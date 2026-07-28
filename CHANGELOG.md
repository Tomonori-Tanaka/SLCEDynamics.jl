# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — the documentation is published

- **<https://tomonori-tanaka.github.io/SLCEDynamics.jl/dev/>** — the Documenter site is
  now deployed to GitHub Pages by the `documentation build` CI job (`deploydocs`
  in `docs/make.jl`, `permissions: contents: write` on the job). It was being
  built on every push and then thrown away.
- **Per-line source links work**: `remotes = nothing` / `edit_link = nothing` are
  gone in favour of the real repository, so every docstring on the site links to
  its own lines on GitHub and each page has an "Edit on GitHub" link.
- README carries a docs badge, a CI badge and the site URL.

### Changed — BREAKING: the family-wide naming batch

Follows SLCE.jl's third naming batch (see its `CHANGELOG`); landed in all four
repositories together.

- **`run_llg_gpu` → `gpu_run_llg`** (exported). SLCEMonteCarlo puts the device
  qualifier in front on all five of its GPU entry points (`gpu_run_sweeps!`,
  `gpu_metropolis_sweep!`, `gpu_displacement_sweep!`, `gpu_energy_gradient!`,
  `gpu_zlm_rows!`) and on every GPU type (`GPUTiledHamiltonian`, `GPUChainState`,
  `GPUGradientScratch`) — as does this package's own `GPULLGState`. This one function
  put it at the end. Nothing but the name changes; **scripts on a cluster that call
  `run_llg_gpu` need the one-word edit**.
- **`resolve_kt` now comes from `SLCE`**, not `SLCEMonteCarlo` — the conversion moved
  upstream to the package all three samplers share. Same function, same values.
- **Prose: `SCE` → `SLCE`, "spin–lattice cluster expansion"** — the ratified name;
  the index page expanded it as "symmetry-adapted cluster expansion", which was wrong.
  Documentary only; no identifier changed.

### Fixed

- **`equilibrium_stats` ignored an `Evaluable`'s `scope`**, passing `n_active` to
  every one of them. Upstream, `scope` exists precisely because a per-site
  quantity is only intensive when divided by the sites that carry it: `:spin`
  (χ, Binder) wants `n_spin_active`, `:energy` (specific heat) wants `n_active`.
  `LLGResult` now carries both counts and the bridge picks between them, as the
  MC side does.

  Not currently reachable, and said plainly rather than dressed up: the two
  counts differ only when displacement-only sites exist, and a joint
  `TiledHamiltonian` dies at the first measurement (the parked gap in
  `CLAUDE.md`). What this removes is a trap laid for whatever lands spin–lattice
  dynamics here — a magnetization normalized by the total active count is wrong
  by the ratio of the two, with nothing to announce it. Gated in
  `test_observables.jl` on a directly-constructed `LLGResult` whose counts
  diverge, since no run can produce one.

### Added

- `test_sqw_core.jl` "series layout is xyz-fastest-then-site": states the recorded
  `:spins` series' on-disk layout against the configuration itself. It closes no gap —
  a transpose is already caught by the analytic ring gates, verified by mutation — but
  it makes a layout change fail at the source rather than as a wrong ring dispersion
  three files downstream. The path round-trip cannot see it: writer and reader move
  together.

- `bench/Project.toml`: this package's own GPU bench environment, carrying CUDA
  (which the shared `@slce` env deliberately does not). The two GPU benches used
  to borrow `SLCEMonteCarlo.jl/bench/gpu` on the cluster, which meant the
  upstream repo's tracked `Project.toml` had to list `SLCEDynamics` — a
  downstream dependency in the upstream repo, present only in the cluster's copy
  and therefore a permanently divergent tracked file there (it broke a job once
  when an rsync overwrote it, and it was the last obstacle to updating the
  cluster by `git pull` instead of rsync). The bench headers and `CLAUDE.md` now
  document the one-time per-machine setup; the `Manifest.toml` stays untracked.

### Changed — BREAKING: the upstream `Observable` contract is now `f(view)`

- `SLCEMonteCarlo.Observable` takes a single `MCView` argument
  (`v.config`, `v.disps`, `v.energy`, `v.H`) instead of `(config, energy, H)`,
  so that the growing sampled state — displacements arrived with the upstream
  spin–lattice channel — does not break every observable each time it grows.
  Every observable definition here and in user code must be rewritten:
  `(cfg, E, H) -> cfg[1][3]` becomes `v -> v.config[1][3]`.
- `_measure!` builds the view with an **empty** `disps`. Spin dynamics has no
  displacement channel, so a displacement observable throws here instead of
  reporting a confident zero. `trajectory_observable` moved with it; the
  recorded `:spins` series layout is unchanged, so stored checkpoints still
  read.

### Changed — BREAKING: package renamed SCESpinDynamics.jl → SLCEDynamics.jl (SLCE family, M0)

- The whole family is renamed to the **spin–lattice cluster expansion (SLCE)**
  stem per `docs/specs/spin-lattice-ce-design.md` §2 (SLCE.jl /
  SLCEMonteCarlo.jl / SLCEDynamics.jl / SLCETools.jl). Package + module name
  changed; **UUID kept** (path-dev Manifests stay resolvable). Old model /
  checkpoint artifacts are unaffected (persistence schemas carry versions,
  not package names).

### Fixed

- Test fixtures, guide pages and `examples/bloch_mt.jl` still used SLCE's old
  `BasisSpec` keyword `isotropy`, which was renamed to `soc` **with its meaning
  inverted** (`isotropy = true` ⇔ `soc = false`). Every `BasisSpec` call here
  therefore raised a `MethodError` on an unknown keyword, so the unit suite
  errored at fixture construction and the `@example` blocks could not run.
  Migrated all 11 sites with the inversion applied.

### Changed

- `run_llg_gpu` is now **exported** (was public-unexported pending the A100
  go/no-go): the GO (25× at l044 8³) and the quantum-device smoke both
  landed. `GPULLGState` stays public-unexported machinery.

### Added

- Quantum-thermostat example tier (decision-record gate G6):
  `examples/bloch_mt.jl` — M(T) of a cubic ferromagnet under both
  thermostats vs linear spin-wave estimates (classical linear deficit,
  quantum Bloch-law flattening) — and the "G6: ring S(q,ω) intensities"
  testset gating per-mode ω-integrated S⁺⁻(q_m) of one quantum spectrum
  (x = ħω/kT ≈ 1.0/2.2/3.5) against the linear-response prediction
  (`_qt_predict_ring`), by both the equal-time observable and the
  Parseval bin-sum route.
- Documenter documentation site (`docs/`): index, getting started, guides
  (deterministic LLG, thermal sLLG, quantum thermostat, S(q,ω),
  checkpoint/resume, GPU) with executed examples, and the API reference
  covering every exported and public name (`make -C docs build`).
- `bench/bench_gpu_qtb.jl`: the quantum-thermostat device smoke (classical
  zero-sized-filter run, quantum end-to-end, device checkpoint/resume
  bitwise gate, classical-vs-quantum step-time readout). A100 smoke PASSED
  2026-07-19 (job 858230, l044 4³): all gates green on CUDA, resume bitwise,
  quantum step-time overhead within timing noise (see
  `docs/specs/quantum-thermostat.md` Q4).
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
