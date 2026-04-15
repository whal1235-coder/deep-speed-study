"""
Step 1: Profile VRAM usage per sample on a single GPU.

Sweeps batch_size from 1 upward, records peak VRAM after a forward+backward
pass, and estimates activation memory per sample.

Usage:
    python profile_vram.py             # uses config.yaml, runs on cuda:0
    python profile_vram.py --device cuda:1
"""

import argparse
import yaml
import torch
import torch.nn as nn

from model import DecoderOnlyTransformer


def measure_vram(model, vocab_size, seq_len, batch_size, device):
    """
    Run one forward + backward pass with the given batch_size.
    Returns peak VRAM allocated (bytes), or None on OOM.
    """
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    src = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    criterion = nn.CrossEntropyLoss()
    try:
        logits = model(src)
        loss   = criterion(logits.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        # clear gradients so they don't accumulate across iterations
        model.zero_grad(set_to_none=True)
        return peak
    except torch.cuda.OutOfMemoryError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_batch", type=int, default=256,
                        help="Upper bound for the batch_size sweep")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device
    print(f"Device : {device}  ({torch.cuda.get_device_name(device)})")
    print(f"Total VRAM : {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB\n")

    model = DecoderOnlyTransformer(
        cfg["vocab_size"], cfg["seq_len"], cfg["d_model"],
        cfg["n_heads"], cfg["n_layers"], cfg["d_ff"], cfg["dropout"]
    ).to(device)

    # ── Baseline: model parameters + buffers only ────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    base_vram = torch.cuda.memory_allocated(device)
    print(f"Model-only VRAM  : {base_vram / 1e6:.1f} MB")

    # ── Sweep batch sizes ────────────────────────────────────────────────────
    print(f"\n{'batch':>6}  {'peak VRAM (MB)':>15}  {'activation MB/sample':>22}")
    print("-" * 50)

    last_ok_batch = None
    last_ok_peak  = None

    batch_size = 1
    while batch_size <= args.max_batch:
        peak = measure_vram(model, cfg["vocab_size"], cfg["seq_len"], batch_size, device)
        if peak is None:
            print(f"{batch_size:>6}  {'OOM':>15}")
            break

        activation_mb_per_sample = (peak - base_vram) / batch_size / 1e6
        print(f"{batch_size:>6}  {peak / 1e6:>15.1f}  {activation_mb_per_sample:>22.2f}")

        last_ok_batch = batch_size
        last_ok_peak  = peak

        # Sweep: 1,2,4,8,… then fine-grained near the limit
        if batch_size < 8:
            batch_size += 1
        elif batch_size < 32:
            batch_size += 4
        else:
            batch_size += 8

    # ── Summary ──────────────────────────────────────────────────────────────
    if last_ok_batch is not None:
        act_per_sample = (last_ok_peak - base_vram) / last_ok_batch / 1e6
        print(f"\n=== Summary ===")
        print(f"Max working batch_size : {last_ok_batch}")
        print(f"Peak VRAM at that batch : {last_ok_peak / 1e6:.1f} MB")
        print(f"Model-only VRAM         : {base_vram / 1e6:.1f} MB")
        print(f"Activation VRAM (total) : {(last_ok_peak - base_vram) / 1e6:.1f} MB")
        print(f"Activation MB / sample  : {act_per_sample:.2f} MB")


if __name__ == "__main__":
    main()
