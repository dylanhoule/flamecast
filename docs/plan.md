# Flamecast: project breakdown

## Context
`prd.md` describes a 3-layer project (GPU Rothermel simulator, procedurally generated dataset, neural surrogate + browser demo). This file turns it into ordered parts with tasks and exit gates. Dates live in `prd.md` (Milestones and timeline); this file only tracks order. Dylan writes the code; Claude reviews, explains, and scaffolds only when asked.

## Scope rules
1. **Gates are hard.** A part's gate must pass before the next part on the same track starts.
2. **Parking lot, not plan.** New ideas go to the bottom of this file, reviewed only after launch.
3. **Shrink inside a part, don't extend it.** If a part runs long, cut within it before pushing later parts.

## Decisions captured
- PACE access: **not confirmed**. Develop T0/T1/dataset locally on Mac (arm64, MPS) at 64 to 128 grids; cloud GPU fallback for T2.
- Env: **uv + Python 3.12** (local default is 3.14, too new for PyTorch wheels).
- Fuels: **full 40-model Scott & Burgan table** implemented; dataset uses **GR2, SH5, TL3** (GR2 dynamic, fixed curing), 4th OOD fuel static (GS2 or TU1).
- Rothermel reference values: **unresolved**, task 1.2.
- **Spec review (Oct 18):**
  - CPU baseline is **multicore Dijkstra**, not NumPy Jacobi; GPU target X set at the T1 gate. Kernel judged on time to finish and work vs Dijkstra, not bandwidth percentage.
  - **16-neighbor stencil default**, no corner cutting; 8 for speed comparison, 32 for accuracy.
  - Determinism comes from **running to convergence**; T0 stays Jacobi for readability, T2 may update in place within tiles.
  - Ellipse shape from **effective wind** (wind + slope), capped. Wind is **midflame, 0 to 6 m/s** (OOD 6 to 10).
  - Problem statement rewritten (world-model data framing; gradients are sparse, not absent).
  - Surrogate: fire-size control and per-bucket IoU, ellipse baseline as input channel, data scaling curve, unseen barrier layouts as a 5th OOD set, soft-min simulator gradients as the inverse-mode baseline.
  - **WebGPU cut from v1**; demo simulator is TypeScript/WebAssembly Dijkstra.
  - **ML runs in parallel with T2**; the dataset only needs T1.
- Tracking: W&B. CI: GitHub Actions for CPU tests, GPU tests manual on PACE. Repo private until launch.
- Name: `flamecast` for now. Write-up: blog post for v1; arXiv-style paper is stretch.

## Status (Oct 23)
- Part 0: done except 0.5 (PACE email not sent).
- Part 1: not started; 1.2 oracle still unresolved.
- Without PACE, everything except Part 3 (CUDA) can run on the Mac.

## Parts (in dependency order)

### Part 0: Setup
- [x] 0.1 `uv init`, Python 3.12, deps: numpy, scipy, torch, pyyaml, zarr, pytest, matplotlib
- [x] 0.2 Repo layout per PRD (create dirs only as each part needs them)
- [x] 0.3 GitHub repo (workflow in place: .github/workflows/ci.yml) + Actions running `pytest -m "not gpu"`
- [x] 0.4 One shared config schema (YAML) + loader: grid size, cell size, fuel table, stencil, max iters (SIM-6)
- [ ] 0.5 Email PACE / Supercomputing Club about access (blocking risk, do day 1)

### Part 1: T0 NumPy reference + CPU baseline
- [ ] 1.1 `sim/fuels.py`: full SB40 table (40 models + non-burnable); dataset uses GR2, SH5, TL3
- [ ] 1.2 Pick a Rothermel oracle (pyretechnics scripted, spot-checked against GTR-371 / BehavePlus); fixtures cover all 40 models
- [ ] 1.3 `sim/rothermel.py`: each term tested separately, then R0 and head rate; imperial inside
- [ ] 1.4 Ellipse: LB from **effective** wind (wind + slope), capped; verify LB wind units; eccentricity, R(θ)
- [ ] 1.5 `sim/reference.py`: 16-neighbor (8, 32 options) Jacobi relaxation, 6 h cap, no corner cutting (diagonal: both corner cells; knight: both crossed cells), raise on non-convergence
- [ ] 1.6 `sim/dijkstra.py`: SciPy Dijkstra on the same graph and edge costs; bit-identical to T0
- [ ] 1.7 `tests/test_physics.py`:
  - spread rate within 5% of the oracle
  - stencil shape error vs exact circle/ellipse at 8/16/32 neighbors, calm and wind on/off axis (16: at most 3% calm)
  - on-axis wind ellipse LB within 5%
  - upslope factor + calm-slope elongation
  - barriers: straight and diagonal 1-cell lines, every stencil
  - exact 90-degree rotation
  - update-order independence (Jacobi, in-place random order, Dijkstra bit-identical)
  - grid convergence along axes only
- **Gate:** all Layer 1 physics tests pass on T0; Dijkstra matches T0 bit for bit.

### Part 2: T1 batched PyTorch + baseline benchmark
- [ ] 2.1 `sim/torch_sim.py`: same math on `[B,H,W]` via shifted-tensor stencil (pad, never roll), CPU, MPS, CUDA
- [ ] 2.2 Convergence check: stop when no cell changes; raise at the iteration limit
- [ ] 2.3 Multicore Dijkstra baseline over the same 10,000 scenarios, graph build included, named CPU
- [ ] 2.4 `tests/test_tiers.py`: T0 vs T1, 100 random scenarios, 1e-4 relative, cells within 1e-3 h of the cap excluded from mask comparison
- [ ] 2.5 Benchmark script (`make bench`): T1 vs Dijkstra; **set the GPU target X**
- **Gate:** tier test passes; baseline and T1 numbers recorded; X set.

### Track A, Part 3: T2 CUDA kernel (hard 3-week timebox)
- [ ] 3.1 PyTorch C++ extension skeleton, naive one-thread-per-cell kernel, batch on grid z
- [ ] 3.2 Match T0 via the tier test (marked `gpu`)
- [ ] 3.3 Shared-memory 32x32 tile + 2-cell halo; iterate in-tile until settled; block-level changed flag
- [ ] 3.4 Active-tile list (skip tiles with no active cells); edge costs computed on the fly
- [ ] 3.5 Report: time to finish batch, updates per fire vs Dijkstra edge count, Nsight roofline placement
- [ ] 3.6 Slurm scripts in `slurm/`
- **Gate:** matches T0 and beats Dijkstra by X, or the timebox ends and v1 continues on T1.

### Track B, Part 4: Dataset (in parallel with Part 3, starts at the T1 gate)
- [ ] 4.1 `data/terrain.py`: seeded fractal heightmaps, noise-shaped fuel patches, 0 to 10% non-burnable, fuel-break line generator (training layouts + unseen OOD layouts)
- [ ] 4.2 `data/sampler.py`: Latin hypercube over moisture, **midflame** wind 0 to 6 m/s (OOD 6 to 10), direction
- [ ] 4.3 Fire-size check: burned-fraction distribution; resample so it spans a range; store fire-size bucket per sample (DATA-8)
- [ ] 4.4 Domain-exit check: how often fires reach the edge before 6 h; pick a fix and record it
- [ ] 4.5 `data/generate.py`: run T1 in chunks, write Zarr (1 scenario per chunk), metadata JSON, folder named by config hash + git commit
- [ ] 4.6 Splits by terrain seed: 8k/1k/1k + 5 OOD x 500 (high wind, high relief, unseen fuel, multi-ignition, unseen barriers; multi-ignition needs SIM-9)
- [ ] 4.7 Byte-identical regeneration test; `make dataset`; viewer script (DATA-6)
- **Gate:** all splits generated and versioned in under 15 min; fire-size distribution reported.

### Track B, Part 5: Surrogate (first U-Net during the T2 timebox)
- [ ] 5.1 `model/baseline_ellipse.py` B0 (effective wind)
- [ ] 5.2 Input channel builder (~13 channels incl. distance transform and B0 arrival map)
- [ ] 5.3 Physics-consistent 8x augmentation (rotate wind + slope vectors) + `tests/test_augment.py`
- [ ] 5.4 `model/unet.py` predicting a correction to B0; arrival + burned-mask heads; masked MSE + BCE
- [ ] 5.5 `model/train.py`: AdamW, cosine + warmup, bf16, early stop on val IoU, W&B logging of config/commit/dataset hash
- [ ] 5.6 Data scaling: train at 1k / 3k / 8k / 50k fires (generate extra train-only scenarios)
- **Gate:** beats B0 in every fire-size bucket and reaches 0.85 IoU at 2 h on validation.

### Buffer: finals

### Part 6: Evaluation + export
- [ ] 6.1 `eval/metrics.py`: IoU at 1/2/4/6 h (overall and per fire-size bucket), arrival MAE, area error, Hausdorff
- [ ] 6.2 `eval/run_eval.py`: all models x all test sets x fire-size buckets, one table; test set touched once; prior work (Burge et al.) in the table after verifying their numbers
- [ ] 6.3 OOD curves (wind, relief, fuel, barriers), `eval/gallery.py` 20 worst cases, data scaling plot
- [ ] 6.4 `model/export_onnx.py`, float16, PyTorch vs ONNX within 1e-3
- **Gate:** results final, ONNX parity passes.

### Part 7: Demo + write-up
- [ ] 7.1 Vite + TypeScript app shell
- [ ] 7.2 TypeScript or WebAssembly Dijkstra solver in a Web Worker at 256 x 256; parity vs T0 on exported JSON fixtures
- [ ] 7.3 Side-by-side panels, ONNX Runtime Web (WebGPU, WASM fallback), lazy-loaded weights
- [ ] 7.4 Controls: midflame wind 0 to 10 m/s (training range 0 to 6 shaded, 20-ft equivalent shown), direction, moisture, ignition click, firebreak brush, time scrubber, new terrain, "Break it" at 9 m/s
- [ ] 7.5 Diff overlay + live IoU/area/timing readout
- [ ] 7.6 Mobile stacked layout; Lighthouse first paint under 3 s on LTE
- [ ] 7.7 Pages: How it works (incl. no-spotting limit), Results, Links; deploy static to a dylanhoule.dev subdomain
- [ ] 7.8 README benchmark table (vs Dijkstra, hardware named), blog post, repo public
- **Gate:** every v1 acceptance box in the PRD ticked.

### Stretch (post-launch)
WebGPU simulator (SIM-11), FNO (ML-6), 3D toggle (DEMO-9), inverse mode with soft-min simulator baseline (ML-8, DEMO-10), autoregressive (ML-7), arXiv paper, real terrain (DATA-7; the full SB40 table makes this physics-free).

## Risks
- **GPU barely beats CPU Dijkstra.** The naive kernel's lead is about 10x at best; work efficiency (3.3, 3.4) is the whole game. If X isn't met, publish the honest number and the work-count analysis.
- **GR2 dynamic loading** complicates Rothermel; if the oracle check fails, swap to static GS2 for v1.
- **No PACE access yet:** line up a cloud GPU for Part 3 if not confirmed before the T1 gate.
- **Surrogate wins on small fires only:** fire-size control (4.3) and per-bucket reporting catch it.

## Parking lot
New ideas go here, not into a part. Reviewed only after launch.

- 
