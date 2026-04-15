"""
Baseline: Decoder-only Transformer training WITHOUT DeepSpeed.
We will use this as a reference to compare against the DeepSpeed version.
"""

import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import DecoderOnlyTransformer


# ─── Prefetcher ──────────────────────────────────────────────────────────────

class Prefetcher:
    """Wraps a DataLoader and copies the next batch to GPU while the
    current step is still computing, hiding HtoD transfer latency."""

    def __init__(self, loader, device: str):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self._prefetch()

    def _prefetch(self):
        try:
            src, tgt = next(self.loader)
        except StopIteration:
            self.next_src = None
            self.next_tgt = None
            return
        # Start HtoD copy on a separate stream so it overlaps with compute
        with torch.cuda.stream(self.stream):
            self.next_src = src.to(self.device, non_blocking=True)
            self.next_tgt = tgt.to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        # Wait until the prefetch stream has finished the copy
        torch.cuda.current_stream().wait_stream(self.stream)
        src, tgt = self.next_src, self.next_tgt
        if src is None:
            raise StopIteration
        self._prefetch()   # kick off copy for the batch after next
        return src, tgt


# ─── Synthetic dataset ────────────────────────────────────────────────────────

class RandomTokenDataset(Dataset):
    """Generates random token sequences on the fly (no real data needed)."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len     = seq_len
        self.vocab_size  = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        # input: tokens [0, seq_len-1]
        # target: same sequence shifted by 1 (next-token prediction)
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return tokens[:-1], tokens[1:]


# ─── Training loop ────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg: dict):
    device = "cuda:0"
    print(f"Device : {device}")

    # Dataset & DataLoader
    dataset    = RandomTokenDataset(cfg["num_steps"] * cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=2, pin_memory=True)
    prefetcher = Prefetcher(dataloader, device)

    # Model, optimizer, loss
    model     = DecoderOnlyTransformer(
        cfg["vocab_size"], cfg["seq_len"], cfg["d_model"],
        cfg["n_heads"], cfg["n_layers"], cfg["d_ff"], cfg["dropout"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    print(f"Parameters : {count_parameters(model):,}")
    print(f"Steps      : {cfg['num_steps']}  |  Batch size: {cfg['batch_size']}\n")

    model.train()
    total_tokens = 0
    t_start      = time.perf_counter()

    for step, (src, tgt) in enumerate(prefetcher):
        if step >= cfg["num_steps"]:
            break


        logits = model(src)                                              # (B, T, V)
        loss   = criterion(logits.view(-1, cfg["vocab_size"]), tgt.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_tokens += cfg["batch_size"] * cfg["seq_len"]

        if (step + 1) % 20 == 0 or step == 0:
            elapsed      = time.perf_counter() - t_start
            tokens_per_s = total_tokens / elapsed
            print(
                f"step {step+1:4d}/{cfg['num_steps']} | "
                f"loss {loss.item():.4f} | "
                f"throughput {tokens_per_s:,.0f} tok/s"
            )

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal time : {elapsed:.1f}s")
    print(f"Avg throughput: {total_tokens / elapsed:,.0f} tok/s")


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
