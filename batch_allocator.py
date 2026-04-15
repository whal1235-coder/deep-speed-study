"""
Batch allocation utilities for heterogeneous multi-GPU training.

Responsibilities are split across three functions:

  allocate_batch   — pure performance-proportional split (Step 3)
  vram_caps        — per-GPU sample capacity from VRAM (Step 2)
  remaining_vram   — VRAM left after model + activations (Step 4)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GPUSpec:
    name:           str
    total_vram_mb:  float   # total physical VRAM in MB
    relative_perf:  float   # relative compute throughput (e.g. 2.5 vs 1.0)


# ── Step 3 ────────────────────────────────────────────────────────────────────

def allocate_batch(
    gpu_specs:   list[GPUSpec],
    total_batch: int,
    caps:        list[int] | None = None,
) -> list[int]:
    """
    Split total_batch across GPUs to minimise per-step wall-clock time.

    Each GPU should receive batch_i proportional to its relative_perf so that
    batch_i / perf_i is equal for all GPUs (all finish at the same time).

    When a GPU cannot take its ideal share (due to an optional VRAM cap),
    its excess is redistributed to the remaining GPUs, still proportionally.

    Parameters
    ----------
    gpu_specs:   GPU descriptors — only relative_perf is used here.
    total_batch: Total number of samples to distribute.
    caps:        Optional per-GPU maximum sample counts (from vram_caps()).
                 If None, no VRAM constraint is applied.

    Returns
    -------
    List of per-GPU batch sizes (same order as gpu_specs).
    """
    n     = len(gpu_specs)
    perfs = [g.relative_perf for g in gpu_specs]
    caps  = caps if caps is not None else [total_batch] * n

    if total_batch > sum(caps):
        raise ValueError(
            f"total_batch={total_batch} exceeds combined GPU capacity={sum(caps)}"
        )

    allocated = [0] * n
    remaining = total_batch
    active    = list(range(n))

    while active and remaining > 0:
        perf_sum = sum(perfs[i] for i in active)

        capped = [
            i for i in active
            if remaining * perfs[i] / perf_sum > caps[i] - allocated[i]
        ]

        if not capped:
            ideal     = {i: remaining * perfs[i] / perf_sum for i in active}
            floor_val = {i: int(ideal[i])                   for i in active}
            leftover  = remaining - sum(floor_val.values())
            # Give leftover samples to GPUs where the time increase is smallest
            priority  = sorted(active, key=lambda i: (floor_val[i] + 1) / perfs[i])
            for k, i in enumerate(priority):
                allocated[i] += floor_val[i] + (1 if k < leftover else 0)
            break

        for i in capped:
            avail         = caps[i] - allocated[i]
            allocated[i] += avail
            remaining     -= avail
        active = [i for i in active if i not in capped]

    return allocated


# ── Step 2 ────────────────────────────────────────────────────────────────────

def vram_caps(
    gpu_specs:         list[GPUSpec],
    model_vram_mb:     float,
    act_per_sample_mb: float,
) -> list[int]:
    """
    Maximum samples each GPU can hold given its VRAM.

    cap_i = floor((total_vram_i - model_vram) / act_per_sample)
    """
    return [
        int((g.total_vram_mb - model_vram_mb) // act_per_sample_mb)
        for g in gpu_specs
    ]


# ── Step 4 ────────────────────────────────────────────────────────────────────

def remaining_vram(
    gpu_specs:         list[GPUSpec],
    batch_per_gpu:     list[int],
    model_vram_mb:     float,
    act_per_sample_mb: float,
) -> list[float]:
    """
    VRAM remaining on each GPU after model parameters and activations.

    remaining_i = total_vram_i - model_vram - batch_i * act_per_sample
    """
    return [
        g.total_vram_mb - model_vram_mb - b * act_per_sample_mb
        for g, b in zip(gpu_specs, batch_per_gpu)
    ]


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gpus = [
        GPUSpec("RTX 3090",    total_vram_mb=24_576, relative_perf=2.5),
        GPUSpec("RTX 2080 Ti", total_vram_mb=11_264, relative_perf=1.0),
    ]
    MODEL_VRAM     = 168.5
    ACT_PER_SAMPLE = 466.0

    caps = vram_caps(gpus, MODEL_VRAM, ACT_PER_SAMPLE)
    print(f"VRAM caps: {[f'{gpus[i].name}={caps[i]}' for i in range(len(gpus))]}\n")

    for tb in [2, 3, 67]:
        alloc = allocate_batch(gpus, tb, caps)
        rem   = remaining_vram(gpus, alloc, MODEL_VRAM, ACT_PER_SAMPLE)
        times = [alloc[i] / gpus[i].relative_perf for i in range(len(gpus))]
        print(f"total_batch={tb}")
        for i, g in enumerate(gpus):
            print(f"  {g.name:<14} batch={alloc[i]:>3}  remaining={rem[i]:>8.1f} MB  time={times[i]:.3f}")
        print()
