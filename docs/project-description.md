# Flamecast

A GPU wildfire simulator that runs 10,000 fire scenarios in one launch, plus a neural network trained on those runs that predicts a fire's full spread in under a second. Successor to [Flames](https://flames.dylanhoule.dev), rebuilt for throughput instead of one interactive fire.

Full spec: `prd.md`. Task tracker: `docs/plan.md`.

## The problem

Simulators are how world-model teams make training data. Flamecast asks three questions on a small, fully controlled problem: how fast a GPU fire simulator can generate data, how well a learned surrogate reproduces it, and where the surrogate breaks. A surrogate also gives smooth gradients for inverse problems (where did this fire start, where should a firebreak go) where the simulator's own gradients are sparse.

What it does not claim: that Rothermel is slow. It is cheap algebra, and a full 256 x 256 fire takes 10 to 20 ms on one CPU core with Dijkstra. The speed claim is made against that baseline.

## The approach

Three layers, each a project on its own:

1. **GPU simulator.** Grid-based Rothermel surface spread, solved as arrival-time relaxation (parallel shortest path). Built in tiers, each validated against the one before:
   - T0: NumPy reference (ground truth), plus a SciPy Dijkstra CPU baseline that matches it bit for bit
   - T1: batched PyTorch tensor ops (CPU, Apple MPS, CUDA); GPU speedup target set here against multicore Dijkstra
   - T2: work-efficient CUDA kernel via a PyTorch C++ extension (active tiles, in-tile iteration, edge costs on the fly; 3-week timebox)
2. **Dataset.** 12,500 procedurally generated scenarios (fractal terrain, patchy fuel maps, fuel breaks, moisture, wind, ignition) sampled with a Latin hypercube, fire sizes controlled, stored in versioned Zarr, split by terrain seed into train, validation, test, and 5 out-of-distribution sets. Generated with T1, so ML work runs in parallel with the CUDA kernel.
3. **Neural surrogate.** A U-Net that maps input maps (elevation, slope, fuel, moisture, wind, ignition, and the ellipse baseline's arrival map) to a correction of that baseline. Scored against the baseline alone, per fire-size bucket, with a data scaling curve (1k to 50k training fires). Exported to ONNX and run in the browser. FNO, an autoregressive model, and inverse mode (with gradients through a soft-min simulator as its baseline) are stretch goals.

The demo shows a live simulator (Dijkstra in TypeScript or WebAssembly) and the neural network side by side on the same fire, with controls for wind, moisture, ignition, firebreak brush, and time, and a "Break it" preset that pushes the model outside its training range.

## Physics in brief

- **Fuels:** full 40-model Scott & Burgan table implemented; the dataset uses GR2 (grass, dynamic, fixed curing), SH5 (shrub), TL3 (timber litter), plus non-burnable. A 4th unseen fuel is held out for OOD testing.
- **Spread rate:** Rothermel head-fire rate with wind and slope factors summed as vectors; imperial units internally.
- **Shape:** Anderson length-to-breadth ellipse from effective wind (wind plus slope), capped, ignition at the rear focus. Wind is midflame wind, 0 to 6 m/s in training.
- **Grid:** 256 x 256 cells at 30 m (7.68 km square), 16-neighbor stencil by default with no corner cutting through firebreaks, 6 hour cap. Any update order run to convergence gives the same bits, so T0 uses Jacobi for readability and T2 can update in place.
- **Out of scope:** crown fire, spotting, fire-atmosphere coupling, suppression. Surface spread only.

## v1 success criteria

| Metric | Target |
| --- | --- |
| Speedup | X times multicore Dijkstra on a named CPU, same 10,000 scenarios; X set at the T1 gate |
| Kernel | Time to finish, updates per fire vs Dijkstra, roofline placement |
| GPU vs reference agreement | Max relative error 1e-4 |
| Stencil shape error | At most 3% on calm flat ground (16 neighbors) |
| Surrogate burned-area IoU at 2 h | 0.85 in-distribution, per fire-size bucket, beating the ellipse baseline in every bucket |
| Data scaling | IoU at 1k / 3k / 8k / 50k training fires |
| Surrogate latency | 10 ms on GPU, 1 s in a laptop browser |
| OOD report | Error curves for wind, relief, fuel, and barrier layouts |
| Demo | Live simulator and surrogate side by side, desktop and phone |

## Validation

Two layers. The simulator is checked against physics: published Rothermel values within 5%, stencil shape error against the exact circle and ellipse, wind ellipse ratio, upslope factor and elongation, diagonal and one-cell barriers, exact 90-degree rotation symmetry, update-order independence, grid convergence along the axes, and tier agreement. The surrogate is checked against the simulator (IoU at 1/2/4/6 h overall and per fire-size bucket, arrival-time MAE, area error, Hausdorff distance), always next to the ellipse baseline and prior work, with the test set touched once and the 20 worst cases published.

## Non-goals

Not an operational fire prediction tool and never claimed to be. No multi-GPU scaling, no real-world data in v1, no speed parity on phones.

## Stack

Python 3.12 (uv), NumPy, SciPy, PyTorch, CUDA C++, Zarr, YAML configs, pytest, Weights & Biases, Slurm on Georgia Tech PACE (cloud GPU fallback). Demo: TypeScript, Vite, a TypeScript/WebAssembly Dijkstra solver, ONNX Runtime Web, static hosting on a dylanhoule.dev subdomain. WebGPU simulator is a post-launch stretch.

## Timeline

| Phase | Dates |
| --- | --- |
| T0 NumPy reference + CPU Dijkstra baseline | Oct 19 to Nov 1, 2026 |
| T1 batched PyTorch; set GPU target vs baseline | Nov 2 to Nov 15 |
| T2 CUDA kernel (timeboxed), with dataset + first U-Net in parallel | Nov 16 to Dec 6 |
| Finals buffer | Dec 7 to Dec 16 |
| U-Net iteration, data scaling, OOD eval, ONNX export | Dec 17 to Jan 3, 2027 |
| Browser demo + write-up | Jan 4 to Jan 20 |
| Launch | Jan 24, 2027 |

## Why it matters

It covers GPU systems, machine learning, and simulation as a data source in one project: a small, honest version of what simulation and world-model teams do at scale. Build a simulator, generate data from it, learn a fast model of it, measure how much data it needs, and publish where it breaks.
