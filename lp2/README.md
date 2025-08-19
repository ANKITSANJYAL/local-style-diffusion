# LPA2 (Local Prompt Adaptation) — Benchmark-Grade Repo

This repo houses a training‑free method to route **style vs. content** prompt tokens to different UNet stages and timesteps in diffusion models (SDXL, SD1.5). It includes one‑command generation, metrics, ablations, human‑study grids, and benchmark adapters.

## Quickstart
```bash
make setup
make run
make eval RUN=<run_id_from_stdout>
