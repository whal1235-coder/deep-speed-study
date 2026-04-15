"""
Decoder-only Transformer training with DeepSpeed ZeRO-3.

Batch allocation
----------------
  With heterogeneous GPUs (3090 x2.5 vs 2080Ti x1.0), each rank receives a
  batch proportional to its relative compute performance so that all GPUs
  finish their forward/backward at the same time (no idle waiting).

  Single GPU: full batch goes to GPU 0.
  Dual GPU  : allocate_batch() splits proportionally (e.g. 32 / 13 for bs=45).

What lives where (with CPU offload)
-------------------------------------
  GPU  : model parameters (gathered per layer), activations, gradients
  CPU  : optimizer states (Adam m/v) + fp32 master weights + parameter storage

Usage
-----
  # single GPU (3090)
  deepspeed --include localhost:0 train_deepspeed.py

  # both GPUs (heterogeneous-aware batch split)
  deepspeed --include localhost:0,1 train_deepspeed.py
"""

import argparse
import json
import os
import time

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import deepspeed

from model import DecoderOnlyTransformer
from batch_allocator import GPUSpec, allocate_batch


# ─── GPU specs (order must match CUDA device indices) ─────────────────────────

_GPU_SPECS = [
    GPUSpec("RTX 3090",    total_vram_mb=24_576, relative_perf=2.5),
    GPUSpec("RTX 2080 Ti", total_vram_mb=11_264, relative_perf=1.0),
]


# ─── Synthetic dataset ────────────────────────────────────────────────────────

class RandomTokenDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len     = seq_len
        self.vocab_size  = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return tokens[:-1], tokens[1:]


# ─── Training loop ────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg: dict, args):
    device = f"cuda:{args.local_rank}"
    torch.cuda.set_device(device)

    # ── Per-rank batch size (performance-proportional split) ──────────────────
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    specs      = _GPU_SPECS[:world_size]
    alloc      = allocate_batch(specs, cfg["batch_size"])
    batch_size = alloc[args.local_rank]

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset    = RandomTokenDataset(cfg["num_steps"] * batch_size,
                                    cfg["seq_len"], cfg["vocab_size"])
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DecoderOnlyTransformer(
        cfg["vocab_size"], cfg["seq_len"], cfg["d_model"],
        cfg["n_heads"], cfg["n_layers"], cfg["d_ff"], cfg["dropout"]
    )

    # ── DeepSpeed config ──────────────────────────────────────────────────────
    with open("ds_config.json") as f:
        ds_config = json.load(f)

    # Each rank reports its own micro-batch size to DeepSpeed.
    ds_config["train_micro_batch_size_per_gpu"] = batch_size

    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    criterion = nn.CrossEntropyLoss()

    if args.local_rank == 0:
        alloc_str = " | ".join(
            f"{specs[i].name}={alloc[i]}" for i in range(world_size)
        )
        print(f"Device     : {device}  ({torch.cuda.get_device_name(device)})")
        print(f"Parameters : {count_parameters(model):,}")
        print(f"Batch alloc: {alloc_str}  (total={cfg['batch_size']})")
        print(f"Steps      : {cfg['num_steps']}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model_engine.train()
    total_tokens = 0
    t_start      = time.perf_counter()

    for step, (src, tgt) in enumerate(dataloader):
        if step >= cfg["num_steps"]:
            break

        src = src.to(device)
        tgt = tgt.to(device)

        logits = model_engine(src)
        loss   = criterion(logits.view(-1, cfg["vocab_size"]), tgt.view(-1))

        model_engine.backward(loss)
        model_engine.step()

        total_tokens += batch_size * cfg["seq_len"]

        if args.local_rank == 0 and ((step + 1) % 20 == 0 or step == 0):
            elapsed      = time.perf_counter() - t_start
            tokens_per_s = total_tokens / elapsed
            print(
                f"step {step+1:4d}/{cfg['num_steps']} | "
                f"loss {loss.item():.4f} | "
                f"throughput {tokens_per_s:,.0f} tok/s"
            )

    if args.local_rank == 0:
        elapsed = time.perf_counter() - t_start
        print(f"\nTotal time    : {elapsed:.1f}s")
        print(f"Avg throughput: {total_tokens / elapsed:,.0f} tok/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    train(cfg, args)
