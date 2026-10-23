# Part 1 guide: T0 NumPy reference

What Part 1 builds, why each task exists, the gotchas, and what to study before and during it.

## Why Part 1 matters

T0 is the ground truth for the whole project. T1, T2, the browser solver, the dataset and the surrogate are all judged by whether they match T0. It only has to be two things: **physically right** (matches published Rothermel numbers) and **readable** (plain NumPy, no speed tricks). If T0 is wrong, every later number is wrong, and the tier tests won't catch it, because they only check that tiers agree with each other.

```
fuel table + moisture ──► Rothermel ──► R0 per cell (no wind, no slope)
wind vector + slope ────► φ_w, φ_s vectors ──► R_head, direction, eccentricity per cell
                                               │
ignition mask ─────────────────────────────────┴──► relaxation ──► arrival time map T
                                                    (Dijkstra on the same graph = CPU baseline, same bits)
```

Suggested order: **1.2 → 1.1 → 1.3 (term by term, each with a test) → 1.4 → 1.5 → 1.6 → 1.7.** Task numbers match `docs/plan.md` Part 1. The oracle comes first because it fixes the curing method, wind limit, and fixture format everything else is checked against.

---

## 1.1 `sim/fuels.py`: fuel table

Implement the **full 40-model Scott & Burgan table** (GR1 to GR9, GS1 to GS4, SH1 to SH9, TU1 to TU5, TL1 to TL9, SB1 to SB4) plus non-burnable. The dataset only uses GR2, SH5, TL3 (and one OOD fuel), but the other rows are just transcription, and they keep a real-terrain preset possible later without touching the physics.

A Scott & Burgan fuel model is a set of fuel-bed parameters:

| Field | Units | Notes |
|---|---|---|
| Loads: 1h, 10h, 100h dead, live herb, live woody | tons/acre | Convert to lb/ft² (×0.04591) |
| SAV ratio σ per class | 1/ft | 10h always 109, 100h always 30, others vary |
| Fuel bed depth δ | ft | |
| Dead moisture of extinction Mx | fraction | Above this, no spread |
| Heat content h | BTU/lb | 8000 for all SB40 |

Constants for every model: particle density ρ_p = 32 lb/ft³, total mineral S_T = 0.0555, effective mineral S_e = 0.010.

Approximate values from memory, **verify against GTR-153**:
- **GR2**: 0.1 t/ac 1h, 1.0 live herb, depth 1.0 ft, Mx 15%. **Dynamic.**
- **SH5**: 3.6 / 2.1 / 0 dead, 2.9 live woody, depth 6.0 ft, Mx 15%. Static.
- **TL3**: 0.5 / 2.2 / 2.8 dead, no live, depth 0.3 ft, Mx 20%. Static.

**Dynamic GR2:** a `curing` fraction of the live herb load moves into a new dead herbaceous class. It keeps the herb SAV (1800) but takes dead fine moisture. The real method derives curing from live herb moisture (30% = fully cured, 120% = uncured). We fix `curing: 0.5` in the config, so the oracle must be run with the same curing value or 1.2 fails for reasons unrelated to bugs.

Non-burnable: no parameters, R0 = 0.

## 1.2 Pick the oracle (do first)

"Within 5% of published values" needs published values. Options:
- **pyretechnics** (Python): Rothermel + SB40 including dynamic models, easy to script over wind × moisture grids.
- **BehavePlus** (GUI) / **firelab/behave** (its C++ library): the most trusted reference, harder to automate.
- **GTR-371 worked examples**: hand-worked cases with **intermediate values** (I_R, ξ, β...). These make per-term tests in 1.3 possible.

Recommendation: pyretechnics as a dev-only tool. Write a one-off script that generates a fixture table (all 40 fuels × wind × moisture → R_head plus intermediate terms), commit it as CSV/JSON, spot-check a few rows against GTR-371 or BehavePlus. Tests read the fixture, so CI never depends on the oracle.

Details where implementations differ (match the oracle on each):
- **Wind limit:** Rothermel capped effective wind at 0.9·I_R. Andrews 2018 discusses dropping or modifying it.
- **Wind height:** Rothermel wants midflame wind. We treat input as midflame and skip the wind adjustment factor. Document it.
- **Live moisture of extinction** is computed: `Mx_live = 2.9·W'·(1 − Mf_dead/Mx_dead) − 0.226`, floored at Mx_dead.

## 1.3 `sim/rothermel.py`: term by term

`R = I_R·ξ·(1+φ_w+φ_s) / (ρ_b·ε·Q_ig)`, where each symbol is its own calculation:

| Term | Meaning | Formula |
|---|---|---|
| β, β_op | packing ratio, optimum | β = Σw₀/(ρ_p·δ), β_op = 3.348·σ^−0.8189 |
| ρ_b | bulk density | Σw₀/δ |
| σ | characteristic SAV, surface-area weighted | weights A_i = σ_i·w₀_i/ρ_p |
| Γ' | reaction velocity | Γ'max·(β/β_op)^A·exp(A(1−β/β_op)), Γ'max = σ^1.5/(495+0.0594σ^1.5), A = 133σ^−0.7913 |
| η_M | moisture damping | 1 − 2.59r + 5.11r² − 3.52r³, r = Mf/Mx clamped to 1 |
| η_s | mineral damping | 0.174·S_e^−0.19 ≈ 0.417 |
| **I_R** | reaction intensity, BTU/ft²/min | Γ'·Σ_cat(w_n·h·η_M·η_s), dead and live separately |
| **ξ** | propagating flux ratio | exp((0.792+0.681σ^0.5)(β+0.1)) / (192+0.2595σ) |
| **ε** | effective heating number | exp(−138/σ_i) per class |
| **Q_ig** | heat of preignition, BTU/lb | 250 + 1116·M_f |
| **φ_w** | wind factor | C·U^B·(β/β_op)^−E; C, B, E functions of σ |
| **φ_s** | slope factor | 5.275·β^−0.3·tan²φ |

The denominator is really `ρ_b·Σ f_i·ε_i·Q_ig,i`, since each class has its own moisture.

**Why test each term:** three 2% bugs give a final R about 6% off with no clue where. Per-term tests against GTR-371 intermediate values point straight at the broken line.

**Imperial inside:** every constant (495, 0.0594, 192) assumes ft, lb, BTU, minutes. Take SI at the boundary (wind m/s ×196.85 → ft/min, cell size m → ft), compute imperial, return ft/min. φ_w wants wind in **ft/min**. Anderson's LB (1.4): **verify the units** (Finney 1998 states m/s; Anderson's original used mi/h; the mi/h reading gives LB ≈ 12 at 10 mi/h, which looks wrong). Mixing them is a classic bug.

## 1.4 Ellipse: direction and shape

Rothermel gives one head-fire rate. The grid needs a rate in every direction.

**Vector sum.** φ_w points downwind, φ_s points up the gradient. Sum them:
- `R_head = R0·(1 + |φ_w⃗ + φ_s⃗|)`
- `θ_max` = direction of the sum

**Effective wind speed.** Slope stretches fires too. Invert φ_w to get the wind that alone produces the combined φ: `U_eff = (φ_total·(β/β_op)^E / C)^(1/B)`. Convert to the LB relation's units, then:
- `LB = 0.936·e^(0.2566U) + 0.461·e^(−0.1548U) − 0.397` (exactly 1.0 at U = 0; that's what the 0.397 is for)
- cap LB at the FARSITE value (high winds otherwise give LB in the tens)
- `e = √(LB²−1)/LB`
- `R(θ) = R_head·(1−e)/(1−e·cosθ)`

This R(θ) is measured from the **rear focus**, where ignition sits. That's exactly what relaxation asks: how fast does fire travel from point j in direction θ. In continuous space the reachable set at time t is that ellipse scaled by t. On the grid it is the convex polygon through t·R(θ_k) for the stencil directions θ_k, at any cell size: an octagon with 8 neighbors, which is why 16 neighbors is the default (see Stencil below).

**Tip:** compute cosθ as a **dot product** of unit vectors, not via `atan2` + `cos`. A 90° rotation only swaps components and flips signs, which IEEE does exactly. atan2/cos can change the last bit and break the exact-rotation test.

## 1.5 `sim/reference.py`: relaxation solver

`T_i = min(T_i, min_j [T_j + d_ij / R_{j→i}])`. Ignition cells start at 0, all others +∞. Repeat until nothing changes.

1. **Precompute travel times once.** Wind is uniform and static, so `R_{j→i}` depends only on cell j and the offset. Build `tt[k] = d_k / R_j(θ_k)` for each of the 16 offsets (d = 1, √2, or √5 cells) before the loop. The loop becomes pure min-plus with no physics. Fine for T0; T2 computes costs on the fly instead of storing 16 floats per cell.
2. **Loop:** for each offset, shift `T` (pad with +∞, use slicing, **not** `np.roll` which wraps around edges), add `tt[k]`, take the min. Jacobi (read old T, write new T) because it's easiest to read. Determinism does **not** come from Jacobi: each update is a min over sums of positive costs and float addition preserves order, so **any** update order run to convergence gives the same bits as Dijkstra (SIM-4). Early stopping breaks this, so convergence is checked.
3. **Non-burnable:** R = 0 → `tt` = +∞, and mask them so they never update. **No corner cutting:** a diagonal move is blocked if either orthogonal corner cell is non-burnable; a knight move if either crossed cell is.
4. **Stop** when `np.array_equal(T_new, T)` (safe, values only decrease) or at `max_iters`.
5. **Cap:** T in minutes; anything above 360 is unburned.

**`max_iters: 500` is suspicious.** The front moves at most one cell per iteration, so you need at least as many iterations as the longest path in cells. A straight run on 256 is ~256 steps; paths around barriers are much longer. If T0 hits max_iters without converging it should **warn or raise**, not return a wrong map. Better: make the limit grid-relative.

## 1.6 `sim/dijkstra.py`: CPU baseline

`scipy.sparse.csgraph.dijkstra` on a CSR graph with the same nodes, the same 16 offsets, the same `tt[k]` weights, and the same blocked moves. Two jobs:
1. **Correctness:** on the same scenario it must match T0 **bit for bit**. If not, the graph or the costs differ, which is a bug in one of them.
2. **Speed baseline:** this, not NumPy Jacobi, is what the GPU must beat. Count graph construction in its time (building the CSR is often slower than the search itself). Run over scenarios on every core in Part 2.

## 1.7 `tests/test_physics.py`: how each check measures

| Test | Measurement | Gotcha |
|---|---|---|
| Spread rate 5% | Loop fixture rows, compare R_head | Unit test on rothermel.py, no grid |
| Stencil shape error | Calm flat, then uniform wind on and off axis, at 8/16/32 neighbors; arrival error vs exact circle/ellipse | Tolerance per stencil (8.2% / 2.7% / 1.3% calm); 16 must be at most 3% |
| Wind ellipse LB 5% | Length = head + back along wind axis; breadth = max width across | Big fire so cell rounding is small; wind on an axis |
| Upslope | No wind, constant tanφ: uphill rate / R0 = 1+φ_s; downhill backs slower | Checks slope-only stretching via U_eff; a circle means LB ignored slope |
| Barrier | Straight 1-cell and diagonal 1-cell lines, every stencil; beyond stays +∞ | Diagonal and knight-move corner cutting |
| Update-order independence | Jacobi, in-place random order, Dijkstra on one scenario | Must be bit-identical |
| 90° rotation | `np.rot90` grids, rotate wind and slope **vectors**, output == rot90(original) **bitwise** | Rotate components, never add 90° to an angle. Slope components swap and flip sign |
| Grid convergence | Same area at 128/256/512 (60/30/15 m), vs analytic ellipse | See below |

**Grid convergence caveat:** stencil direction bias does **not** shrink with refinement. 8-neighbor is ~8% off at 22.5° at any cell size. Refinement only shrinks ignition-size and terrain-sampling error. Measure convergence **along the axes**; treat off-axis bias as a fixed floor that the stencil choice sets (the shape-error test covers it). A naive all-directions test will flatline and look like a bug.

## Stencil: why 16 neighbors (part of 1.5)

With 8 directions, reaching angle θ (0° to 45°) costs path length `cosθ + (√2−1)·sinθ`. At 22.5° that's **1.082**: 8.2% late, radius ~7.6% short. Octagonal fronts, well past 3%.

Adding knight moves (±1,±2), (±2,±1) with d = √5·cell gives directions at 0°, 26.57°, 45°... Worst gap 26.57°, worst error `1/cos(13.28°) − 1` ≈ **2.7%**. Passes 3%, barely, which is why the PRD threshold is exactly 3%. 32 neighbors (adding (±1,±3), (±3,±1), (±2,±3), (±3,±2)) is the accuracy check.

**Big gotcha:** a knight move (0,0) → (2,1) **jumps over** column 1. A 1-cell firebreak drawn in the demo would leak. Block a move if any cell it passes through is non-burnable (two cells for knight moves; for diagonals decide whether one or both orthogonal neighbors block). Test 1-cell straight and diagonal barriers under both stencils. The GPU halo becomes 2 cells (Part 3 already plans for it).

## Gate

All Layer 1 physics tests pass on T0 with the default 16-neighbor stencil, and Dijkstra matches T0 bit for bit.

## Decisions to make before coding

1. Wind limit: match the oracle.
2. Wind input is midflame (no adjustment factor), 0 to 6 m/s in the dataset.
3. LB wind units and cap value.
4. Corner-cutting rule: both corner cells for diagonals, both crossed cells for knight moves.
5. Grid-relative `max_iters`, raise when hit.
6. Grid convergence measured along axes only.
7. Coordinate conventions (see below), written down once in `sim/` and used everywhere.

---

# What to research and learn

Ordered by when you need it. **Must** = blocks a task. **Should** = prevents a known bug. **Optional** = deeper understanding.

## A. Rothermel physics (before 1.2 / 1.3)

- **Must: Andrews 2018, RMRS-GTR-371**, "The Rothermel surface fire spread model and associated developments." The primary reference. Read: model inputs, the full equation set, the worked examples (your per-term test values), the wind limit discussion, and dynamic fuel models. [PDF](https://d10.nrfirescience.org/sites/default/files/2023-07/rmrs_gtr371.pdf)
- **Must: Scott & Burgan 2005, RMRS-GTR-153**, "Standard fire behavior fuel models." The SB40 tables for GR2/SH5/TL3, and how dynamic curing works.
- **Should: Rothermel 1972, INT-115.** The original paper. Skim for intuition on what each term physically means (heat source vs heat sink).
- **Should: Albini 1976, INT-30.** Corrections to Rothermel's original constants (A exponent, multi-class weighting). Explains why implementations differ.
- **Should: Andrews 2012, RMRS-GTR-266**, on midflame wind and the wind adjustment factor. You're skipping WAF, so know what you're skipping.

Concepts to be able to explain in one sentence each: packing ratio and why there's an optimum; why moisture damping is a cubic; effective heating number (why fine fuels matter most); heat of preignition; dead vs live moisture of extinction; head, flank and backing fire; midflame vs 20-ft wind.

## B. Fire shape (before 1.4)

- **Must: Anderson 1983, INT-305**, "Predicting wind-driven wild land fire size and shape." Source of the LB relation.
- **Must: FARSITE elliptical dimensions docs** (linked in the PRD appendix), for the 0.397 adjusted form and focus-based R(θ).
- **Should: Finney 1998, RMRS-RP-4**, FARSITE technical reference. How wind and slope vectors combine and how effective wind speed is derived.
- **Optional: Richards 1990**, elliptical front propagation (Huygens' principle) for the math behind why ellipses compose.

## C. Arrival time as a shortest-path problem (before 1.5)

- **Must: Bellman-Ford and Dijkstra.** The relaxation is literally parallel Bellman-Ford on a grid graph. Understand why it converges, why the number of iterations equals the longest path in hops, why any update order reaches the same answer, and why Dijkstra does far less work serially (it's your CPU baseline) but doesn't parallelize.
- **Must: Finney 2002**, "Fire growth using minimum travel time methods" (Canadian Journal of Forest Research). The same idea (MTT) used in FlamMap. Closest prior art to what you're building.
- **Should: the eikonal equation and fast marching** (Sethian's level set book or any lecture notes). Arrival time solves an anisotropic eikonal equation; relaxation is a discrete approximation.
- **Should: chamfer distance transforms (Borgefors 1986).** The exact analysis of 8- and 16-neighbor distance error. This is where the 8.2% and 2.7% numbers come from.
- **Optional: Jacobi vs Gauss-Seidel iteration** in general, and why for shortest paths in-place updates only change how fast you converge, not where.

## D. Coordinates and conventions (before 1.4, Should)

The most likely source of silent bugs. Decide and document:
- Row/col vs x/y: does +row mean south? Does `np.gradient` return (d/drow, d/dcol)? (It does, in axis order.)
- Wind direction convention: meteorological wind is where it blows **from**; spread goes **toward** the opposite. Pick one for the config.
- Slope: gradient points uphill; aspect conventions vary. Store gradient components (as the PRD says), not angles.
- `np.rot90` rotates counterclockwise in the (row, col) plane: work out by hand how a wind vector and a gradient vector must transform to match.

## E. NumPy and floating point (during 1.5 / 1.6)

- **Must:** slicing-based shifts with `np.pad(..., constant_values=np.inf)`; why `np.roll` is wrong here (wraps edges).
- **Must:** `inf` arithmetic (`inf + x = inf`, `x / 0` warnings, `np.errstate`).
- **Should:** float32 vs float64 choice for T0 (float64 for the reference is fine; tier tests compare at 1e-4 relative).
- **Should:** IEEE basics: which ops are exact (sign flips, swaps, `min`, commutative add of two numbers), which aren't (transcendentals, reassociated sums). Needed for the bitwise rotation test.

## F. Testing (during 1.6)

- **Must:** `pytest.mark.parametrize` (fuel × wind × moisture rows), `pytest.approx(rel=0.05)`, custom markers (`gpu`, `slow`) registered in `pyproject.toml`.
- **Should:** fixture/golden files: generate once with a script, commit, never regenerate silently.
- **Optional:** property-based testing (Hypothesis) for "rotation exact for any random scenario."

## G. Tools to try hands-on

- **pyretechnics**: install in a throwaway env, reproduce one GTR-371 example, then script the fixture table.
- **BehavePlus**: run 2 or 3 cases by hand to cross-check pyretechnics on GR2 dynamic loading in particular.
- **scipy.sparse.csgraph.dijkstra**: build a tiny 5x5 graph by hand and check it against your Jacobi before writing 1.6.

## Self-check before starting 1.3

You're ready when you can answer, without notes:
1. Why does R0 get computed once per scenario but R(θ) per direction?
2. What does `curing = 0.5` do to GR2's fuel classes?
3. Why does slope change the ellipse shape, not just the head rate?
4. Why does an 8-neighbor stencil give octagons, and why doesn't refinement fix it?
5. Why do Jacobi, in-place updates, and Dijkstra all give the same bits, and when would they stop agreeing?
6. How do wind and gradient vectors transform under `np.rot90`?
