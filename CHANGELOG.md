# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
