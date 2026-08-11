# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — S(q,ω) subtracts the mean of the ANALYZED span (review 2026-08-11 M6)

**Changes S(q,ω) numbers whenever the trimmed window is not a power of two.**
The per-site mean (and hence the elastic tensor and the subtracted DC) was
computed over the whole trimmed window `L`, while the Welch segments transform
only `(nsegments−1)·hop + M ≤ L` samples — as little as half of `L` on the
default path (`M = prevpow(2, L)`). The residual per-site DC `ē_L − ē_span`
landed in the **inelastic** ω = 0 bin, defeating the stated invariant that
Bragg weight is separated into `S_el`; Parseval and the channel sum rule are
structurally blind to it. Mean, `S_el`, and the projection loop now all run
over the analyzed span. Gate: samples beyond the span are junk-invariant
(`test_sqw_core.jl`).

### Fixed — `equilibrium_stats` refuses evaluables on a `b_ext ≠ 0` run (review 2026-08-11 M5)

The evaluables read the recorded `:energy` series — the SLCE energy alone (the
`SLCEMonteCarlo` observable contract) — while a field-carrying run samples
`exp(−(E_SCE + E_Zeeman)/kT)`: specific heat from `var(E)/kT²` silently omitted
the Zeeman channel (essentially all of C for a weakly anisotropic spin in a
strong field), and the zero-field susceptibility formula is not `∂m/∂B` at
finite field. `LLGResult` now records `b_ext` (new field; positional
constructors gain one argument), and `equilibrium_stats` refuses nonempty
evaluables on such a run unless `allow_evaluables = true` — mirroring the
quantum-thermostat refusal. Raw time-average stats remain available, and
`result.energies` (Zeeman-inclusive) supports hand-built estimates.

### Fixed — config doors validate inactive columns too (review 2026-08-11)

The energy path evaluates the harmonic kernels at every site, so an
out-of-domain inactive `config0` placeholder died as a bare Legendre
`DomainError` naming nothing — the audit-#1 failure shape surviving on the
inactive path (`SLCEMonteCarlo` refuses the same matrix loudly). Both doors now
validate every column: `_config_projected` passes inactive columns through the
non-projecting `Trusted` door (bitwise, as the resume gates require), and
`_config_verbatim` validates all columns on restore.

### Changed — aliasing screen per q + structural floor; spin-side masks; SEM; off-grid drop (review 2026-08-11)

- The temporal-aliasing screen now (i) screens **per q** (worst q wins) — the
  all-q aggregate diluted a fold confined to part of a long `q_path` below any
  threshold; (ii) floors the threshold at twice the actual edge-bin fraction —
  at `nfft = 8` the discrete edge band holds 12.5 % of a flat spectrum, above
  the 5 % continuum ceiling the 0.10 was calibrated against, so the screen
  false-alarmed structurally there; (iii) warns instead of silently passing
  when the spectral weight is not finite (`NaN > threshold` is `false`).
- Every spin-side site mask reads `site_has_spin` instead of `site_active`
  (integrators, noise, thermostat, renormalization, means, S(q,ω) site lists,
  GPU mask upload). Bitwise no-op today — the two predicates coincide on every
  Hamiltonian this package can run — but displacement-only sites would have
  been precessed and renormalized the day spin–lattice dynamics lands.
- The ensemble SEM accumulates scatter about the fixed reference `S_1` instead
  of the one-pass `Σ|S|²/R − |S̄|²` form, which cancelled to exactly `0.0`
  below ~1e-8 relative scatter (algebraically identical otherwise).
- `trajectory`'s off-grid final-measurement drop derives from the stored
  schedule (`nsteps % measure_interval`), not from float spacing of the
  recorded time grid.

### Added — a temporal-aliasing screen on `structure_factor` (audit #8)

The analysis Nyquist is `π/(measure_interval·dt)` — `measure_interval` times
lower than the integration Nyquist — and spectral content above it folds back
into the band as a spurious branch while every sum rule still passes (aliasing
conserves power), so nothing warned. `structure_factor` now warns when more
than 10 % of the inelastic weight sits in the top 5 % of the frequency range: a
band top crossing the Nyquist folds back continuously, so it necessarily
deposits weight at the edge. The threshold must clear the structureless
ceiling — a perfectly flat spectrum already puts ~5 % of its weight in a 5 %
edge band, and a short heavily-damped thermal run measured 3.0–3.6 % from its
noise floor alone — so 0.10 sits 2× above that ceiling and 10× below the
detected signal (ring fixture, `window = :none`: healthy 3.2e-11, in-band
leakage 8.1e-6, just-past-Nyquist 0.99999). One direction only (docstring
warns): a deeply folded isolated mode lands mid-spectrum (measured 3.6e-7),
where the screen cannot see it. Gates: a synthetic single-bin spiral on the
exact analysis grid, healthy (bin 38, no warning) and folded ("bin 131" →
|ω| = 0.977π, warning).

### Fixed — `_measure!` names the ncomp = 1 mismatch (audit #9)

An `Observable` declared `ncomp == 1` whose function returned a 1-element
vector died as a bare `MethodError` on the column assignment, while the
`ncomp > 1` branch two lines below raised a named `DimensionMismatch`. Both
branches now raise the named error.

### Changed — `config0` enters through the family's unit-direction door

**Breaking for bit-level trajectory comparisons.** `run_llg` and `gpu_run_llg`
now pass every *active* column of `config0` through `SLCE.UnitVector3` — the
family's projecting door: finite components, `|‖e‖ − 1| ≤ 1e-6` (was a local
`< 1e-8`), the component bound `max|component| ≤ 1` asked of the projected
value, and then **exact projection onto the sphere** before the first step.
This fixes a live defect (audit 2026-08-01 #1, rank 2): a near-pole column
`5e-9` off unit cleared the old band unprojected and threw a bare `DomainError`
from `LegendrePolynomials.dnPl` (domain `|z| ≤ 1`) inside the first gradient
evaluation, in both drivers (the check was a verbatim duplicate; it is now one
shared door, `_config_projected`). Costs and non-costs:

- **Trajectories from the same `config0` can differ from pre-change runs by the
  entry projection**: `v/‖v‖` is not bitwise idempotent (~38 % of already-unit
  columns move by ≤ 4.4e-16), and LLG is chaotic. Statistical results are
  unaffected; recorded bit-level pins that start from a raw `config0` must be
  recaptured. Within-suite gates (resume ≡ uninterrupted, determinism, CPU/GPU
  references) all pass through the same door and stay bit-consistent.
- A rounded-decimal direction (e.g. a 4-decimal MAGMOM, ~2e-5 off unit) is
  still refused — the band is `1e-6`; past it, normalize deliberately in your
  own code.
- Inactive sites' entries stay **unvalidated placeholders** and pass through
  bitwise (the frozen-spin and resume contracts rely on this).
- `LLGResult.config` of an `nsteps = 0` run is the door-projected state, not
  the raw argument.

Checkpoint restore is the other half and does NOT project: `_config_verbatim`
now validates active columns through `SLCE.UnitVector3(…, Trusted())` — the
non-projecting door — so a corrupted or hand-edited `state/config` is refused
loudly instead of integrating garbage, while a legitimate file is restored
bit-exactly (resume bit-identity gates unchanged; the Trusted refusal can only
fire on states the uninterrupted run would have killed at its next gradient
evaluation anyway).

### Fixed — the quantum thermostat's stationary initialization was wrong by up to +325 %

**Breaking for seeded quantum trajectories**: `L` changes, so a `QuantumThermostat`
run at a given seed produces a different (correct) trajectory. Classical runs are
byte-identical — `_resolve_quantum_fstate` returns early on `ClassicalThermostat`,
so the classical path cannot reach any of this. Existing checkpoints stay valid:
schema v3 stores the filter *coefficients* and resume never rebuilds `L`.

`_stationary_cov` solved the discrete Lyapunov equation `P = A P Aᵀ + B Bᵀ` at
Float64 and `_stationary_sqrt` took a clamped eigendecomposition square root. The
returned `P` satisfied the equation to 1e-16 while **not being positive
semi-definite**, and the thermal-noise power `h·P·hᵀ + d²` was wrong by more than
1 % over **18.7 %** of the accepted τ range — worst case **+325 %** at
τ = 1.70e-4. Confirmed end to end on a 216-site simple-cubic run: the step-1
thermal-field variance came out **+127 % = +22.9σ** against an independent
oracle. The error decays as `A^k E A^kᵀ` with a half-life of 1e5–5e5 steps, i.e.
hundreds of ps against a spin relaxation time of a few ps, so a run faithfully
follows a bath at the wrong temperature for its whole useful length.

The equation and the assembly were correct: solved at 512 bits the covariance is
strictly positive definite at every accepted τ. The problem is conditioning —
`cond(I − A⊗A)` reaches 2–5e16, past `1/eps`. Two factors multiply: the bilinear
map collapses the poles onto `z = 1` as `1 − ρ(A) = 7.32e-3·τ`, and the DF2T
realization's non-normality `κ(V) ≈ 454/τ` is **squared** by the Kronecker
product. LAPACK balancing was measured and moves `cond` by nothing; a diagonal
similarity cannot remove non-normality.

Now `_stationary_factor` solves in **128-bit precision** and returns a **Cholesky
factor** directly. Measured: strictly positive eigenvalues at every τ (3.6e-21 at
the bound, matching the 512-bit reference), and the variance agrees with an
independent contour-integral oracle to that oracle's own resolution. Solving for
a factor makes PSD **structural** — there is no negative eigenvalue to clamp —
and the factor is unique given positive diagonals, which retires the eigenvector
sign-canonicalization the old form needed because LAPACK does not pin its sign
choice. `_stationary_sqrt` is gone; `_filter_state_space` and `_stationary_cov`
are now parametric in the working precision (Float64 default, bit-identical for
existing callers). Cost: ~8 ms once per run.

**Why nothing caught it.** The F4 gate compared a stream started from `x = L·ζ`
against `dot(h, P*h) + d²` — both sides read the same wrong `L`, so it passed
self-consistently. A reference sharing the core routine is not an oracle
(`~/Packages/CLAUDE.md`, Testing). F3/F4 also ran only at τ ≈ 0.0152, where the
error is 1.6e-15. New gates: **Q-F6** compares the variance against
`(1/2π)∮|H_d(z)|² dθ` computed from the stored coefficients alone — no state
space, no Lyapunov, no `L` — on a fixed stratified τ grid; **Q-F7** scans PSD-ness
across the range (necessary but *not* sufficient: at the worst τ the old
eigenvalue ratio was −1.3e-14 and looked healthy); **Q-F8** shows the precision is
converged rather than pinned. An empirical oracle is impossible here — the
filter's memory is ~1e6 steps, so a burn-in measurement would give σ ≈ 45 %.

### Changed — the τ lower bound is derived, and its rejection message was backwards

`_QT_MIN_TAU` is now `√(eps/(α₀_min · _QT_DC_TOL))` from the shipped sections
(1.0175e-4, i.e. the old hardcoded 1e-4), so a re-fit of `_QT_S_BIQUADS` moves it
instead of silently invalidating it. The *value* was fine; the *rationale* named
the Jury margin, which is a different and much later failure (τ ≈ 1.02e-6,
verified — a section is genuinely rejected as unstable there). What actually
degrades first is the discrete DC gain, a cancellation of order `α₀τ²`, and the
prediction `eps/(α₀τ²)` matches measurement to three digits.

The rejection message claimed "at such τ every physical mode is frozen (ħω ≫ kT);
increase dt". That is backwards: a small τ means a **fine** dt, so the Nyquist
band `x ≤ π/τ` resolves *more* of θ(x), and "increase dt" degrades the
integrator. It now says this is a coefficient-arithmetic limit, not a physical
one.

### Changed — internal names spelled out (no public surface touched)

The `STYLE_GUIDE.md` §1 naming contract's safe tier, applied: internal locals and
private helper functions now spell their words out. Nothing exported, nothing in the
`public` tier, no struct field and no persisted key changed, so this is invisible to
every caller and to every file on disk; the suites are green at the same counts.

Locals: `sc` → `scratch` and `spec` → `run_spec` (both were §1.5 collisions — those
two names belong to SLCEMonteCarlo's `SweepScratch` and SLCE's `BasisSpec`), `res` →
`result`, `ck` → `checkpointer`, `tr` → `trace`, `filt` → `noise_filter`, `bq` →
`biquad`, `ebar` → `spin_means`. Helpers: `_ck_llg!`/`_ck_due` →
`_checkpoint_llg!`/`_checkpoint_due`. `G`, `gzee`, `gth` and `pref` are unchanged:
`gzee` is a field of the exported `LLGProblem`, and the four are one family.

`STYLE_GUIDE.md` §1.9 records what was renamed and what deliberately was not.

### Fixed — `sqw_plusminus` returned an all-`NaN` spectrum for a zero axis

`axis / norm(axis)` on a zero vector propagated `NaN` through the frame construction and
into every element of the returned matrix. `sqw_perp` makes its degenerate case loud;
this one now does too.

### Changed — `equilibrium_stats` documents which site count an evaluable receives

The docstring said `f(means, kT, n)` receives "its active-site count"; the code passes
`n_active` for `scope = :energy` and `n_spin_active` otherwise. The two coincide on a
pure-spin model and diverge exactly where a joint model has displacement-only sites, so
a `:spin`-scoped evaluable written against the documented normalization would have been
wrong by `n_active / n_spin_active` the first time it met one.

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
