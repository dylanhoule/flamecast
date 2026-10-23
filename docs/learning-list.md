# Learning list: non-programming knowledge for Flamecast

What to learn outside of code to build, defend, and write up this project. Ordered roughly by when you need it.

Tags: **[P#]** = plan part where it first matters. **Must** = blocks correct work. **Should** = you'll need it to explain or debug. **Optional** = depth for the write-up or stretch goals.

Part 1 specifics (exact papers per task, self-check questions) live in `part1-guide.md`. This list points there instead of repeating it.

---

## 1. Wildfire basics (start here, no background assumed)

- [ ] **Must [P1] The fire behavior triangle: fuel, weather, topography.** Every input channel in the sim and surrogate maps to one corner. Source: NWCG **S-190** *Introduction to Wildland Fire Behavior* (free online course).
- [ ] **Must [P1] Parts of a fire: head, flanks, back (heel), perimeter, fire front.** The ellipse model and the demo panels use these words. Source: S-190, unit on fire behavior terms.
- [ ] **Must [P1] Surface vs crown vs ground fire.** Rothermel only models surface fire. You need to say this in the limitations section. Source: S-190; NWCG *Glossary of Wildland Fire*.
- [ ] **Should [P1] Spotting and crown fire initiation (what you are not modeling).** The "Break it" demo will invite questions about embers jumping firebreaks. Source: Van Wagner 1977, "Conditions for the start and spread of crown fire" (Can. J. For. Res.); NWCG **S-290** *Intermediate Wildland Fire Behavior*.
- [ ] **Should [P1] Fire behavior outputs: rate of spread, fireline intensity (Byram), flame length.** ROS is what you compute; intensity and flame length are what firefighters use, and are a cheap add to the demo readout. Source: Andrews 2018 GTR-371 (Byram section); S-290.
- [ ] **Should [P1] Units used in the field: chains/hour, ft/min, tons/acre, mph at 20 ft.** Published tables and BehavePlus use these. 1 ch/h = 1.1 ft/min. Source: BehavePlus help; GTR-371 appendix.

## 2. Fuels

- [ ] **Must [P1] What a fuel model is:** a stylized fuel bed (loads by size class, SAV, depth, moisture of extinction, heat content), not a vegetation type. Source: Scott & Burgan 2005, **RMRS-GTR-153**, intro.
- [ ] **Must [P1] Fuel size classes and timelag (1 h, 10 h, 100 h, 1000 h).** Why fine dead fuels dominate spread, and why they react to weather within an hour. Source: GTR-153; S-190 fuels unit.
- [ ] **Must [P1] Dead vs live fuels and moisture of extinction.** Explains why live fuel can stop spread and why Rothermel has two Mx values. Source: GTR-371.
- [ ] **Must [P1] Dynamic fuel models and curing (GR2).** Live herb load transfers to dead as grass cures. Your `curing: 0.5` config is a simplification of this. Source: GTR-153, "dynamic fuel models" section.
- [ ] **Must [P1] Your fuel models in plain words:** GR2 (low load dry-climate grass), SH5 (high load dry-climate shrub), TL3 (moderate load conifer litter), plus the 4th OOD pick (GS2 or TU1). Be able to describe what each looks like on the ground and its expected ROS range. Source: GTR-153 per-model pages (they include photos and ROS/flame length charts).
- [ ] **Should [P1] The original 13 Anderson models vs SB40.** Many papers and tools still use the 13. Source: Anderson 1982, **INT-122** *Aids to Determining Fuel Models*.
- [ ] **Optional [Stretch DATA-7] LANDFIRE fuel rasters (FBFM40).** How real fuel maps are made and their resolution (30 m), for real-terrain mode. Source: landfire.gov product docs.

## 3. Weather and fuel moisture

- [ ] **Must [P1, P4] Dead fuel moisture: equilibrium moisture content, how RH and temperature drive it, diurnal cycle.** Your sampler ranges for moisture need to be physically plausible. Source: S-190 weather unit; Nelson 2000 dead fuel moisture model (Can. J. For. Res.), skim.
- [ ] **Must [P1] Live fuel moisture and seasonal greening/curing.** Sets the live moisture range for SH5 and GR2. Source: GTR-153; NFDRS documentation.
- [ ] **Must [P1] 20-ft wind vs 10 m wind vs midflame wind, and the wind adjustment factor.** You skip WAF; know what your "wind speed" input means and how it relates to a weather forecast. Source: Andrews 2012, **RMRS-GTR-266**.
- [ ] **Must [P1] Wind direction convention (from vs toward).** Meteorological wind is where it comes from. Source: any meteorology intro; `part1-guide.md` section D.
- [ ] **Should [P4] Realistic ranges for the Latin hypercube:** typical and extreme values of 1 h moisture (2 to 30%), live moisture, and 20-ft wind. Source: S-290 fire weather unit; NWCG *Fireline Handbook* / *Incident Response Pocket Guide* fire behavior tables.
- [ ] **Optional [P5] Atmospheric stability, fire-atmosphere coupling, plume-dominated fires.** Why steady-wind Rothermel fails on big fires. Good honest limitations paragraph. Source: S-290; Coen's work on WRF-SFIRE (skim abstract-level).

## 4. Topography and terrain

- [ ] **Must [P1] Slope, aspect, and why fire runs uphill** (preheating of upslope fuel). Source: S-190 topography unit.
- [ ] **Must [P1] Slope as rise/run (tan) vs percent vs degrees.** Rothermel's φ_s uses tan²; mixing these is a classic bug. Source: GTR-371.
- [ ] **Should [P1] Aspect effects on fuel moisture** (south faces drier in the northern hemisphere). You don't model it, but someone will ask. Source: S-190.
- [ ] **Should [P4] Digital elevation models: resolution, units, how gradients are computed from a DEM.** Source: USGS 3DEP overview; any GIS intro on slope from DEM (Horn's method).
- [ ] **Should [P4] What real terrain statistics look like** (relief per km, roughness), so procedural terrain isn't obviously fake and the "high relief" OOD set is meaningful. Source: compare a few 3DEP tiles against your generator output by eye.

## 5. Fire spread modeling (the field you're joining)

- [ ] **Must [P1] Rothermel model, term by term.** See `part1-guide.md` section A.
- [ ] **Must [P1] Elliptical fire shape and length-to-breadth ratio.** See `part1-guide.md` section B.
- [ ] **Must [P1] Minimum travel time (MTT) and its relation to your relaxation.** See `part1-guide.md` section C.
- [ ] **Should [P1] Landscape of operational tools: BehavePlus (point), FARSITE and FlamMap (landscape), Prometheus / Canadian FBP system, ELMFIRE, Cell2Fire.** Your write-up needs a "related work" paragraph and an answer to "why not just use FARSITE". Source: each tool's docs page; Finney 1998 RMRS-RP-4 for FARSITE.
- [ ] **Should [P1] Empirical vs semi-empirical vs physics-based models** (Rothermel vs FDS/WFDS, FIRETEC). Positions your project on the fidelity/speed axis. Source: Sullivan 2009, "Wildland surface fire spread modelling, 1990 to 2007" (3-part review, Int. J. Wildland Fire), skim part 1 intro of each.
- [ ] **Should [P1] Known Rothermel limitations and biases** (tends to underpredict in some shrub types, wind limit controversy, no spotting). Source: GTR-371 discussion sections; Andrews, Cruz & Rothermel 2013 on the wind limit (Int. J. Wildland Fire).
- [ ] **Optional [P8] Model validation in fire science:** how people compare predicted vs observed perimeters (and why it's hard). Source: Cruz & Alexander 2013, "Uncertainty associated with model predictions of surface and crown fire rates of spread" (Env. Modelling & Software).

## 6. Applied math

- [ ] **Must [P1] Shortest paths, Bellman-Ford convergence, eikonal equation, chamfer error.** See `part1-guide.md` section C.
- [ ] **Must [P1] Huygens' principle for front propagation.** Why each cell spreading as an ellipse produces the whole fire. Source: Richards 1990 (Int. J. Numerical Methods in Eng.), or the FARSITE technical reference.
- [ ] **Must [P5] Symmetry groups: the dihedral group D4 (8 rotations/reflections) and how vectors transform under it.** Your 8x augmentation is only "physics-consistent" if wind and slope vectors transform correctly. Source: any intro group theory chapter; for ML framing, Cohen & Welling 2016, "Group equivariant convolutional networks" (sections 1 to 3).
- [ ] **Should [P4] Latin hypercube sampling vs random vs Sobol.** Why LHS covers parameter space better at small N. Source: McKay, Beckman & Conover 1979 (Technometrics); scipy.stats.qmc docs.
- [ ] **Should [P4] Procedural terrain math: fractional Brownian motion, spectral synthesis, Perlin/simplex noise, Hurst exponent.** Controls how rough the terrain is. Source: Musgrave's chapters in *Texturing and Modeling: A Procedural Approach* (Ebert et al.); Red Blob Games "Making maps with noise" (free).
- [ ] **Should [P6] Shape comparison metrics: IoU, Hausdorff distance, area error, MAE on arrival time.** Know what each rewards and punishes (Hausdorff is a worst-case metric, IoU ignores timing). Source: any segmentation evaluation survey; Taha & Hanbury 2015, "Metrics for evaluating 3D medical image segmentation" (BMC Medical Imaging), section on metric properties.
- [ ] **Should [P1, P2] Floating point error and why tiers only agree to 1e-4.** Source: Goldberg 1991, "What every computer scientist should know about floating-point arithmetic," sections 1 and 2.
- [ ] **Optional [P1] Level set methods and fast marching.** Alternative way to solve the same problem. Source: Sethian, *Level Set Methods and Fast Marching Methods*, ch. 1 to 2.

## 7. ML and research method

- [ ] **Must [P5] What a surrogate (emulator) model is and when it's worth it.** Your pitch is "fast approximation of a slow simulator"; know the trade-offs and the "why not just make the simulator faster" counterargument (which T2 itself raises). Source: Kennedy & O'Hagan 2001 (classic emulator framing, intro only); Kasim et al. 2021, "Building high accuracy emulators for scientific simulations with deep neural architecture search" (intro).
- [ ] **Must [P5] Prior ML work on fire spread.** Source: Hodges & Lattimer 2019, "Wildland fire spread modeling using convolutional neural networks" (Fire Technology); Huot et al. 2022, "Next Day Wildfire Spread" dataset (IEEE TGRS); Burge et al. 2023 on recurrent CNN fire spread emulation (Fire Technology). Verify citations when you write the related-work section.
- [ ] **Must [P4, P6] Data leakage and split design.** Why splitting by terrain seed matters, and why the test set is touched once. Source: Kapoor & Narayanan 2023, "Leakage and the reproducibility crisis in ML-based science" (Patterns).
- [ ] **Must [P6] Out-of-distribution evaluation:** interpolation vs extrapolation, and how to present degradation curves honestly. Source: Koh et al. 2021, "WILDS" benchmark paper, intro and section 2.
- [ ] **Must [P5] Baselines and ablations.** Why B0 (pure ellipse) exists and what a reviewer expects next to a headline number. Source: Lipton & Steinhardt 2018, "Troubling trends in machine learning scholarship."
- [ ] **Should [P5] Physics-informed ML vs physics-generated data.** You train on sim output, not a PDE loss. Be able to say why. Source: Karniadakis et al. 2021, "Physics-informed machine learning" (Nature Reviews Physics), sections 1 to 2.
- [ ] **Should [P6] Reporting variance:** multiple seeds, confidence intervals, bootstrap over test scenarios. Source: Bouthillier et al. 2021, "Accounting for variance in machine learning benchmarks" (MLSys).
- [ ] **Should [P8] Responsible framing:** this is a research demo, not an operational tool. Write the disclaimer and the "what it can't do" section with real reasons from sections 1 to 5. Source: Mitchell et al. 2019, "Model cards for model reporting" (use the structure, not the template).
- [ ] **Optional [Stretch ML-6] Neural operators.** Source: Li et al. 2020, "Fourier neural operator for parametric PDEs."

## 8. GPU hardware concepts (theory, not CUDA syntax)

- [ ] **Must [P3] Memory hierarchy: registers, shared memory, L2, DRAM, and their bandwidth/latency.** Your stencil is memory-bound; this is the whole optimization story. Source: Hwu, Kirk & El Hajj, *Programming Massively Parallel Processors* (4th ed.), ch. 4 to 5.
- [ ] **Must [P3] Roofline model and arithmetic intensity.** Your "60% of peak bandwidth" gate is a roofline claim. Source: Williams, Waterman & Patterson 2009, "Roofline: an insightful visual performance model" (CACM).
- [ ] **Must [P3] Stencil tiling and halos.** Why 32x32 tiles plus halo cut DRAM traffic. Source: *Programming Massively Parallel Processors*, stencil chapter.
- [ ] **Should [P3] Coalescing, occupancy, warp divergence.** What Nsight Compute will report and what the numbers mean. Source: NVIDIA CUDA C++ Best Practices Guide, "Performance metrics" and "Memory optimizations" chapters.
- [ ] **Should [P2, P3] Honest benchmarking:** warmup, synchronization, fair CPU baseline, reporting medians. Source: Hoefler & Belli 2015, "Scientific benchmarking of parallel computing systems" (SC15).
- [ ] **Should [P2, P7] Apple Silicon unified memory vs discrete GPU, and browser GPU limits (WebGPU workgroup sizes, fp16 support).** Explains why MPS and WebGPU numbers differ from CUDA. Source: WebGPU spec "limits" section; Apple Metal feature set tables.

---

## Suggested order

1. Week 1: S-190 (sections 1, 4), GTR-153 intro + your 3 fuel pages, then `part1-guide.md` sections A to D.
2. During Part 2 (before Part 3): section 8.
3. Before Part 4: sections 3 (ranges), 4, and the LHS / terrain items in 6.
4. Before Part 5: section 7 Musts, D4 symmetry.
5. Before Part 8: related work and responsible framing items, Sullivan review.
