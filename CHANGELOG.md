# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Quantum thermostat, milestone 1 of 4 (wiring tier; decision record
  `docs/specs/quantum-thermostat.md`): `thermostat` kwarg on `run_llg`
  (`ClassicalThermostat()` default / `QuantumThermostat()`), the DF2T
  biquad-cascade colored-noise machinery
  (`ColoredNoiseFilter`, state-space twin + Lyapunov stationary init,
  step-0 slots-≥2 counter extension — classical counters byte-identical,
  now pinned by literal Philox words), `LLGResult.thermostat`, the
  `τ = kT·dt/ħ ≤ 0.1` validity guard, and `equilibrium_stats(...;
  allow_evaluables)` refusing Boltzmann fluctuation evaluables on quantum
  runs. (The filter was an identity placeholder within this milestone —
  superseded by the milestone-4 constants below.)
- Quantum thermostat, milestone 2 (checkpoint schema v3): `run/thermostat`
  (always written), quantum-only `run/filter_id` (provenance),
  `run/filter/coeffs` (authoritative — resume rebuilds the recurrence from
  the stored coefficients verbatim, never from current package constants)
  and `state/filter` (bitwise restore, the `_config_verbatim` twin);
  v1/v2 files back-read as classical; quantum checkpoint/crash-resume/
  extension are bit-identical.
- Quantum thermostat, milestone 4 (pinned constants + physics gates —
  **the feature is complete; `ClassicalThermostat`/`QuantumThermostat` are
  now exported**): the Barker–Bauer fit ships as 4 DC-normalized s-domain
  biquads (`dev/fit_qtb_filter.jl`, a committed deterministic AAA + LM
  pipeline; max 0.41% relative PSD error over the occupied band, H(0) = 1
  exact, provenance `_QT_FILTER_ID = "bb-aaa10-lm-v1"`), bilinear-mapped
  per run from (kT, dt). Gates: the F1–F4 deterministic filter certificate
  against the shipped constants, and dynamics gates G1–G5 against the exact
  linear-response integral of the shipped filter's own discrete PSD —
  Larmor occupation (the α-broadening is asserted), Einstein specific-heat
  suppression (c < 0.6 kB at x₀ = 3), classical recovery, dimer two-mode
  occupations, and the ≥ 5σ MC-mismatch tripwire documenting that classical
  cross-checks do not apply to quantum runs.
- Quantum thermostat, milestone 3 (GPU): `run_llg_gpu` takes the same
  `thermostat` kwarg; the device noise kernel is a literal port of the
  host cascade on a transposed (coalesced) `n × 6NS` state matrix with
  host-side stationary init uploaded once; filter state joins the
  event-gated checkpoint downloads; the GPU `resume` restores quantum
  files (stored coefficients + state, verbatim). KA-CPU bitwise gates:
  kernel ≡ host fill (nontrivial filter, gth + full state), device
  identity wiring gate, GPU quantum checkpoint/resume/extension.

- GPU LLG/sLLG (`run_llg_gpu`, public unexported pending the A100 go/no-go;
  decision record `docs/specs/gpu-llg.md`): both integrators on a
  KernelAbstractions backend over SCEMonteCarlo's new device gradient
  (`gpu_energy_gradient!`, its G7). The noise stream is the same stateless
  Philox `(seed, site, step)` as the CPU path (same-seed runs are one
  realization across backends); measurements/checkpoints run on host snapshots
  downloaded only at measurement/checkpoint events, so results, statistics,
  S(q,ω), and checkpoint files are identical in kind. Determinism: bitwise for
  fixed (seed, backend, workgroupsize); CI gates the whole driver bitwise
  against a composite keyed reference on the KA-CPU backend. Checkpoint schema
  v2 records (compute, backend, workgroupsize), back-reads v1; the new
  `resume(path, prob, gH)` method continues/extends bit-identically on the
  same backend+ws and refuses compute switches without
  `allow_compute_switch = true`. `LLGResult` gains a `compute` provenance
  field. A100 go/no-go (2026-07-19, kugui job 858227): **GO** — l044 8³ LLG
  step 178.9 ms on the device vs 4471 ms same-node CPU-8T (**25.0×** against
  the ≥ 5× bar); smoke green (repeat identity bitwise, same-seed CPU-vs-GPU
  20-step deviation 7.6e-16).
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
