# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- S(q,ω): the classical dynamical spin structure factor from recorded
  trajectories. `trajectory_observable(H)` records the full configuration
  through the ordinary observable machinery (inheriting cadence, checkpoint
  persistence — a checkpoint file doubles as the trajectory file — and
  bit-reproducibility); `structure_factor(...)` (trajectory / `LLGResult` /
  seed-ensemble / checkpoint-path methods) returns an `SQWResult` holding the
  full 3×3 Hermitian tensor on a two-sided ω axis [rad/fs, + meV axis], with
  the elastic tensor separated and per-site means always subtracted.
  Reductions `sqw_diag`/`sqw_trace`/`sqw_perp`/`sqw_plusminus`/`sqw_elastic`,
  `q_path` with loud commensurate snapping, Welch segmenting, realization
  standard errors (≥ 3 seeds), and an own deterministic radix-2 FFT (no FFTW
  dependency; bit-identical for any `ntasks`). Conventions frozen by exact
  gates: Larmor closed-form single-bin spectrum, the dimer's conserved-q null +
  rigid-rotation mode, the 4-site ring's exact spiral dispersion (±q asymmetry
  pins the spatial sign), Parseval and channel-level sum rules, and an
  ordering pin against `supercell_crystal`. SCEFitting became a genuine src
  dependency (`Crystal`/`n_atoms`/`lattice.reciprocal` for q-space geometry).
- JLD2 checkpoint/resume: `run_llg(...; checkpoint, checkpoint_interval)` writes
  plain-data, atomically-replaced restart files (schema v1, model fingerprint via
  the new `SCEMonteCarlo.model_fingerprint` facade); `resume(path, prob)` — a
  method of `SCEMonteCarlo.resume`, re-exported — continues bit-identically to
  the uninterrupted run (the stateless noise needs no stored RNG state), returns
  a completed file's `LLGResult` without stepping, and can *extend* a run via
  `nsteps` (bit-identical to a single longer run; refused when a completed
  off-grid final measurement would break trace prefixing). The configuration is
  restored verbatim — never through `from_matrix`, whose renormalization would
  ULP-perturb chaotic trajectories.
- Stochastic LLG: `run_llg(...; temperature/kT, seed)` adds the thermal field
  `G_th = σ·ξ`, `σ = √(2 α kB T ħ magmom/(g Δt))` (the FDT amplitude of the
  (1+α²)-prefactored parametrization — no `(1+α²)`), one draw per site and step
  shared by both Heun stages (Stratonovich). Draws are keyed philox4x32-10 via
  SCEMonteCarlo's public facade, counter-tagged `"SD"` in word 4 (disjoint from
  MC streams), stateless — bit-reproducible for any `ntasks` from `(seed, site,
  step)` alone. Gates: single-spin Boltzmann vs analytic quadrature at α = 1.0
  and 0.5 (~0.1σ agreement, α-independent), and sLLG ≡ `run_mc` Metropolis
  equilibrium on the dimer (3σ, τ_int-aware errors both sides).
- `equilibrium_stats(res; evaluables, discard, nbins)`: long-time averages of a
  thermostatted run through SCEMonteCarlo's public binning machinery — raw
  observables get τ_int-aware `ObservableStat`s and the same `Evaluable`
  definitions as the MC drivers (specific heat &c.) are jackknifed with the
  run's `kT` and active-site count. `Evaluable` is re-exported alongside
  `Observable`.
- v0 deterministic LLG slice: `LLGProblem` (per-site `magmom`/`alpha`/`g`,
  uniform `b_ext` [T], prefactor and Zeeman gradient resolved at construction),
  `run_llg` fixed-step driver with stride measurements (`LLGResult`), and two
  integrators — `DepondtMertens` (rotation-based Heun, norm-exact, default) and
  `HeunProjected` (independent cross-check). RNG-free, bit-reproducible for any
  `ntasks`. Validation gates: analytic single-spin Larmor (exact per step for a
  constant field) + damped spiral with observed order 2, two-spin Heisenberg
  uniform rotation about the conserved total spin, α = 0 energy drift O(dt²) on
  an anisotropic model, strict dissipation at α > 0, Depondt↔Heun
  cross-agreement, and bitwise determinism.
- User-defined observables: `run_llg(...; observables)` accepts the same
  `SCEMonteCarlo.Observable(name, ncomp, f)` definitions as the MC drivers
  (including `standard_observables`), fed the SCE energy per that contract, and
  records per-name `ncomp × n_measurements` time-series matrices in
  `LLGResult.series`. Measurements happen at step 0, every `measure_interval`,
  and always at the final step, so the last column always describes the
  returned `config`.
