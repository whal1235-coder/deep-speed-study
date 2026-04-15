# Study For Deep Speed

This project is for trying DeepSpeed library.

The testing model is decoder-only Transformer.

The baseline with single GPU (RTX3090) is in train_baseline.py

We have two GPU (RTX3090 24GB and RTX2080ti 11GB)

The computation performance of RTX 3090 is x2.5 compared to RTX2080ti

The multi-gpu parallel training strategy is:

1. Guess approximate the VRAM usage per activations per sample by increasing batch_size in single GPU
2. Start from batch_size = 1 increasing.
3. Split the batch according to the performance (minimize the expected time for computation)
4. Try ZeRO 3 Offload
5. Increase batch_size and repeat.