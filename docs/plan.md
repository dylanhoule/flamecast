# Flamecast: project breakdown

## Context
`prd.md` describes a 3-layer project (GPU Rothermel simulator, procedurally generated dataset, neural surrogate + browser demo). The repo is greenfield (only `prd.md`). Goal of this step: turn the PRD into an ordered set of parts with tasks and exit gates, written to `docs/plan.md` as a checkbox tracker. Dylan writes the code; Claude reviews, explains, and scaffolds only when asked.

## Decisions captured from Q&A
- PACE access: **not confirmed**. Develop T0/T1/dataset locally on Mac (arm64, MPS) at 64 to 128 grids; cloud GPU fallback for T2 if PACE slips past Nov 1.
- Env: **uv + Python 3.12** (local default is 3.14, too new for PyTorch wheels).
- Fuels: **Scott & Burgan GR2, SH5, TL3**. Note: GR2 is a *dynamic* model (live herbaceous load transfers to dead by curing), so T0 needs a curing-fraction input or a fixed curing value. 4th OOD fuel: pick a static one (e.g. GS2 or TU1) later.
- Rothermel reference values: **unresolved**, becomes task 1.2.
- Demo sim: **WebGPU port in v1** (SIM-11 moves P2 to P0). Launch date: **decide at the T2 gate**, tracked against Jan 24.
- Flames code: not available, so DEMO-9 (3D toggle) is from scratch; keep it P1.
- Tracking: W&B (default, free for students). CI: GitHub + Actions for CPU tests, GPU tests manual on PACE. Repo can stay private until launch.
- Name undecided: use `flamecast` as repo/package name for now.
- Write-up: blog post for v1; arXiv-style paper becomes a stretch item.

## Parts (in dependency order)

### Part 0: Setup (week of Oct 19, ~3 hrs)
- [x] 0.1 `uv init`, Python 3.12, deps: numpy, scipy, torch, pyyaml, zarr, pytest, matplotlib
- [x] 0.2 Repo layout per PRD (create dirs only as each part needs them)
- [ ] 0.3 GitHub repo (workflow in place: .github/workflows/ci.yml) + Actions running `pytest -m "not gpu"`
- [x] 0.4 One shared config schema (YAML) + loader: grid size, cell size, fuel table, stencil, max iters (SIM-6)
- [ ] 0.5 Email PACE / Supercomputing Club about access (blocking risk, do day 1)

### Part 1: T0 NumPy reference (Oct 19 to Nov 1)
- [ ] 1.1 `sim/fuels.py`: SB40 params for GR2, SH5, TL3 (+ non-burnable)
- [ ] 1.2 Pick a Rothermel oracle: BehavePlus tables, or a Python implementation (e.g. pyretechnics) cross-checked against a few published GTR-371 numbers
- [ ] 1.3 `sim/rothermel.py`: each term (I_R, ξ, ρ_b, ε, Q_ig, φ_w, φ_s) tested separately, then R0 and head rate; imperial inside
- [ ] 1.4 Ellipse: Anderson LB, eccentricity, R(θ); wind + slope vector sum for max-spread direction
- [ ] 1.5 `sim/reference.py`: 8-neighbor Jacobi arrival-time relaxation, 6 h cap, non-burnable blocking
- [ ] 1.6 `tests/test_physics.py`: spread rate within 5%, calm-flat circle, wind ellipse LB within 5%, upslope factor, barrier, 90-degree rotation exactness, grid convergence
- [ ] 1.7 16-neighbor stencil (SIM-8) if calm-flat radius variation > 3% (it will be with 8 neighbors; plan for it)
- **Gate:** all Layer 1 physics tests pass on T0.

### Part 2: T1 batched PyTorch (Nov 2 to Nov 15)
- [ ] 2.1 `sim/torch_sim.py`: same math on `[B,H,W]` via shifted-tensor stencil (roll/pad), runs on CPU, MPS, CUDA
- [ ] 2.2 Convergence check: stop when no cell changes, or max iters
- [ ] 2.3 Multicore NumPy CPU baseline (multiprocessing over scenarios) for honest speedup
- [ ] 2.4 `tests/test_tiers.py`: T0 vs T1, 100 random scenarios, 1e-4 relative
- [ ] 2.5 First benchmark script (`make bench`), record numbers
- **Gate:** tier test passes; first speedup number recorded.

### Part 3: T2 CUDA kernel (Nov 16 to Dec 6, hard 3-week timebox)
- [ ] 3.1 PyTorch C++ extension skeleton (`load_inline` or `setup.py`), naive one-thread-per-cell kernel, batch on grid z
- [ ] 3.2 Match T0 via the same tier test (marked `gpu`)
- [ ] 3.3 Shared-memory 32x32 tile + halo, block-level changed flag
- [ ] 3.4 Nsight Compute: bandwidth vs peak, roofline note for the write-up
- [ ] 3.5 Slurm scripts in `slurm/`
- **Gate:** matches T0 and hits 60% peak BW, or timebox ends and v1 continues on T1. **Set the launch date here.**

### Buffer: finals (Dec 7 to Dec 16)

### Part 4: Dataset (Dec 17 to ~Dec 23)
- [ ] 4.1 `data/terrain.py`: seeded fractal heightmaps, noise-shaped fuel patches, 0 to 10% non-burnable
- [ ] 4.2 `data/sampler.py`: Latin hypercube (`scipy.stats.qmc.LatinHypercube`) over moisture, wind speed, direction
- [ ] 4.3 `data/generate.py`: run sim in chunks, write Zarr (1 scenario per chunk), metadata JSON, folder named by config hash + git commit
- [ ] 4.4 Splits by terrain seed: 8k/1k/1k + 4 OOD x 500 (high wind, high relief, unseen fuel, multi-ignition; multi-ignition needs SIM-9)
- [ ] 4.5 Byte-identical regeneration test; `make dataset`
- [ ] 4.6 Viewer script (DATA-6)
- **Gate:** all splits generated and versioned in under 15 min.

### Part 5: Surrogate (~Dec 23 to Dec 31)
- [ ] 5.1 `model/baseline_ellipse.py` B0
- [ ] 5.2 Input channel builder (~12 channels incl. distance transform)
- [ ] 5.3 Physics-consistent 8x augmentation (rotate wind + slope vectors) + `tests/test_augment.py`
- [ ] 5.4 `model/unet.py` with arrival + burned-mask heads; masked MSE + BCE
- [ ] 5.5 `model/train.py`: AdamW, cosine + warmup, bf16, early stop on val IoU, W&B logging of config/commit/dataset hash
- **Gate:** beats B0 and reaches 0.85 IoU at 2 h on validation.

### Part 6: Evaluation + export (Jan 1 to Jan 10)
- [ ] 6.1 `eval/metrics.py`: IoU at 1/2/4/6 h, arrival MAE, area error, Hausdorff
- [ ] 6.2 `eval/run_eval.py`: all models x all test sets, one table; test set touched once
- [ ] 6.3 OOD curves (wind, relief, fuel minimum), `eval/gallery.py` 20 worst cases
- [ ] 6.4 `model/export_onnx.py`, float16, PyTorch vs ONNX within 1e-3
- **Gate:** results final, ONNX parity passes.

### Part 7: WebGPU simulator (new in v1, ~2 weeks, can start in parallel with Part 6)
- [ ] 7.1 Vite + TypeScript app shell
- [ ] 7.2 WGSL compute shader port of the relaxation (fp32), precompute R0 on CPU or shader
- [ ] 7.3 Parity test vs T0 at 128x128 (export a few T0 outputs as JSON fixtures)
- [ ] 7.4 Fallback: CPU Web Worker at 128x128 if WebGPU unavailable (phones)
- **Gate:** matches T0 at 128 and runs 256 live in Chrome.

### Part 8: Demo + write-up (after Parts 6 and 7)
- [ ] 8.1 Side-by-side panels, ONNX Runtime Web (WebGPU, WASM fallback), lazy-loaded weights
- [ ] 8.2 Controls: wind (training range shaded), direction, moisture, ignition click, firebreak brush, time scrubber, new terrain, "Break it"
- [ ] 8.3 Diff overlay + live IoU/area/timing readout
- [ ] 8.4 Mobile stacked layout; Lighthouse first paint under 3 s on LTE
- [ ] 8.5 Pages: How it works, Results, Links; deploy static to a dylanhoule.dev subdomain
- [ ] 8.6 README benchmark table, blog post, repo public
- **Gate:** every v1 acceptance box in the PRD ticked.

### Stretch (post-launch)
FNO (ML-6), 3D toggle (DEMO-9), inverse mode (ML-8, DEMO-10), autoregressive (ML-7), arXiv paper, real terrain (DATA-7).

## Risks newly raised by these answers
- WebGPU in v1 + no Flames reuse adds ~2 to 3 weeks; launch likely mid-Feb. Decide at the T2 gate.
- GR2 dynamic loading complicates Rothermel; if the oracle check fails, swap to static GS2 for v1.
- No PACE access yet: if not confirmed by Nov 1, line up a cloud GPU (Colab Pro / Lambda) for Part 3.

