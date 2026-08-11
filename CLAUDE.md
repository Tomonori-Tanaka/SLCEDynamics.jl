# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

**Before writing, reviewing, or renaming code here, read
[`STYLE_GUIDE.md`](STYLE_GUIDE.md).** Its §1 is the SLCE-family naming contract —
mirrored verbatim in all five packages, canonical copy in `SLCE.jl` — and the
sections after it are this package's own deltas. **Read
[`ARCHITECTURE.md`](ARCHITECTURE.md)** when you need the dependency graph, the
include layering, or a reading order through the source.

## Project goal

Atomistic spin dynamics (LLG / stochastic LLG) on fitted SLCE models, on top of
`SLCEMonteCarlo.jl` (`TiledHamiltonian` + `energy_gradient!`). Priority: physical
correctness of the equation of motion and thermostat (validated against analytic
solutions and, for finite T, against `SLCEMonteCarlo` equilibrium averages) and
the ecosystem's bit-reproducibility discipline. See `SPEC.md` for the working
equation, unit system, and the settled stochastic-LLG design.

## Numerical / physics conventions

- **Units**: energy in the model's units (eV), time in **fs** (`HBAR_EV_FS`),
  external field in **tesla** only at the boundary (`MU_B_EV_T`); μ_B cancels in
  the core evolution (`p_i = g_i/(ħ·magmom_i·(1+α_i²))` carries only ħ and
  g/magmom). Temperature (when the thermostat lands): reuse
  `SLCE.resolve_kt` — `temperature`[K] XOR `kT`[eV], never both.
- **Spins are unit vectors** (`SpinConfig = Vector{SVector{3,Float64}}`);
  `magmom` [μ_B] is a separate per-site parameter (an explicit `LLGProblem`
  argument — a fitted `SLCEModel` does not carry per-atom moments).
- **Sign conventions**: torque `τ_i = −e_i×G_i = m_i×B_eff,i` (physical /
  Landau–Lifshitz, the ecosystem convention); rotation vector
  `ω = −p(G + α e×G)`; dissipation `dE/dt = −Σ p α|G_⊥|² ≤ 0`. γ > 0; a spin at
  +x in B ∥ +z moves toward +y. Larmor anchor: g = 2, 1 T → 27.9925 GHz.
- **Zeeman gradient is NOT tangent-projected** (deliberate — makes constant-field
  Depondt rotation exact); the SLCE gradient arrives projected from
  `energy_gradient!`. Both are valid discretizations; do not "fix" either way
  without moving the single-spin exactness gate.

## Coupled ("linked") code sites — change one, check all

- **`_omega` / `_step!` (both integrators) ↔ the working equation in `SPEC.md` ↔
  the analytic gates**: `test_single_spin.jl` (Larmor exactness, damped spiral,
  order 2) and `test_two_spin.jl` (uniform rotation about the conserved total
  spin) pin the sign, prefactor, and stage structure. Change any of `_omega`,
  `_rotate`, or a stage and re-run both.
- **`LLGProblem` prefactor/Zeeman resolution ↔ `total_energy(::LLGProblem, …)`**:
  `gzee = ∂E_Z/∂e` is used by BOTH the dynamics and the energy; change one side
  and the conservation gate (`test_conservation.jl`) breaks.
- **Inactive-site convention ↔ `SLCEMonteCarlo`'s frozen-spin convention**:
  integrator stages skip, no Zeeman, `mean_spins` excludes, `pref = 0`, spins
  bitwise frozen (`test_single_spin.jl` frozen testset). Mirrors the sibling —
  if `SLCEMonteCarlo` changes `site_active` semantics, this package follows.
- **Upstream gradient contract**: `energy_gradient!` (tangent-projected, exact,
  bit-identical for any ntasks, ≈ 1 sweep/call) is pinned in SLCEMonteCarlo's
  `test_gradient.jl`; this package's determinism gates assume it.
- **FDT constant ↔ `_omega`'s (1+α²) prefactor ↔ the α-independence gate**:
  `σ_i = √(2 α kB T ħ magmom/(g Δt))` in `noise.jl` is correct ONLY for noise
  entering the (1+α²)-prefactored equation that `_omega` implements — change
  either parametrization and the other (and design-notes in SPEC.md) must move;
  the runtime tripwire is the α = 0.5 vs 1.0 Boltzmann gate in
  `test_thermostat.jl`. The same draw must feed BOTH integrator stages
  (Stratonovich) — `_fill_noise!` runs once per step, never per stage.
- **Noise counter layout ↔ SLCEMonteCarlo's philox contract**: word 4 carries the
  nonzero `_DOMAIN_SD` tag (MC streams use 0 — the documented upstream
  contract); `philox_block`/`philox_normal2` are the public facade pinned by the
  Random123 known-answer test upstream. Change the layout and seeded
  trajectories change (breaking-note territory, the P6 scope).
- **Quantum-thermostat cascade ↔ slot map ↔ state-space twin**
  (`thermostat.jl`, `noise.jl` header, `docs/specs/quantum-thermostat.md`
  Q2/Q3): `_qt_cascade!`/`_fill_noise_quantum!` share the classical slots-0/1
  draws (byte-compat of classical mode is pinned by literal Philox words in
  `test_quantum_thermostat.jl`) and claim step 0, slots ≥ 2 for the
  stationary init; `_filter_state_space` is the algebraic twin of the DF2T
  recurrence (the Lyapunov init and the equivalence test derive from it), and it
  is parametric in the working precision because `_stationary_factor` needs the
  SAME assembly at 128 bits — never write a second copy of that loop.
  **The stationary init must be solved in extended precision and returned as a
  factor**: at Float64 the Lyapunov solution is not positive semi-definite
  (`cond(I − A⊗A)` 2–5e16, from `1 − ρ(A) = 7.32e-3·τ` times a SQUARED DF2T
  non-normality — balancing does nothing) and the thermal-noise power was wrong by
  up to +325 %, measured +22.9σ end to end. Its gate must be an oracle the
  implementation cannot reach — the contour integral over the stored coefficients
  (Q-F6) — because the natural-looking check, comparing a stream started from
  `x = L·ζ` against `dot(h, P*h) + d²`, has the same `L` on both sides and passes
  while `L` is wrong. A PSD scan alone is not enough either: at the worst τ the
  eigenvalue ratio read −1.3e-14 and looked healthy.
  Change the recurrence, the lane layout `(c−1)·2NS + 2(j−1) + r`, or the
  slot map and the twin, the tests, and the decision record move together —
  plus the checkpoint `state/filter` layout (schema v3, verbatim restore
  from the STORED coefficients) and the device port
  `_noise_kernel_quantum!`/`_qt_cascade_dev!` (`gpu/kernels.jl` — identical
  expression order on a transposed `n × 6NS` state, bitwise-gated vs the
  host fill in test_gpu_llg.jl a6).
- **Checkpoint writer ↔ reader ↔ the absolute-step purity of `_llg_loop!`**
  (`checkpoint.jl`, `run.jl`): resume bit-identity rests on every per-step
  effect being a pure function of the absolute step index — the Philox counter
  (`_noise_ctrs(site, step)`), the renorm cadence (`step % renorm_interval`),
  and the measurement grid (`step % mi`, plus the final step). Introduce ANY
  carried-over per-step state (an accumulated time, a cached field, a stage
  seed) and the checkpoint schema must grow to carry it (schema-version bump).
  The configuration is restored **verbatim** (`_config_verbatim`) — never
  `from_matrix`, whose renormalization ULP-perturbs chaotic trajectories (the
  bug the crash-resume gate in `test_checkpoint.jl` caught; verified by
  mutation 2026-08-11 — a normalize there turns exactly the resume
  bit-identity gates red, 39 tests, and nothing else). Restore still
  VALIDATES active columns, through `SLCE.UnitVector3(…, Trusted())` — the
  family's non-projecting door — so a corrupted `state/config` is refused
  loudly; that refusal cannot fork a viable trajectory (an off-band or
  `|z| > 1` stored column is one the uninterrupted run would have killed at
  its next gradient evaluation). The model identity
  is `SLCEMonteCarlo.model_fingerprint` (upstream public facade — its mixing is
  part of this file format too).
- **`config0` entry door ↔ SLCE's unit-direction rule ↔ the GPU composite
  reference** (`run.jl` `_config_projected` ↔ SLCE `direction.jl`
  `UnitVector3` ↔ `test_gpu_llg.jl` `_ref_gpu_loop`): both drivers share ONE
  validate-then-project door for `config0` — active columns through the
  projecting `UnitVector3` (finite, `1e-6` band, component bound of the
  projected value), inactive placeholders verbatim. The old local `< 1e-8`
  band without projection was the audited live bug (a near-pole column `5e-9`
  off unit → bare `DomainError` from `dnPl` inside the first gradient
  evaluation), duplicated verbatim in `gpu/run.jl`. Because entry projection
  is bit-non-neutral (~38 % of already-unit columns move by ≤ 4.4e-16), an
  independent bitwise test reference must start from the DOOR-PROJECTED
  config, not the raw fixture — `_ref_gpu_loop` (a3) does. The restore-side
  twin is `_config_verbatim` (previous bullet), which validates without
  projecting — never swap the two doors.
- **GPU stage kernels ↔ the host `_omega`/`_rotate`/`_fill_noise!`/
  `_renormalize_active!` ↔ the composite keyed reference** (`src/gpu/kernels.jl`
  ↔ `integrators.jl`/`noise.jl` ↔ `test_gpu_llg.jl`'s `_ref_gpu_step!`): the
  device kernels are literal expression-order ports; change the host expression
  and the kernels AND the test reference move together (bitwise gates on the
  KA-CPU backend). The noise kernel's inactive branch writes an exact
  `zero(SVector)` — `σ·ξ` at `σ = 0` would emit −0.0 (D12,
  `docs/specs/gpu-llg.md`). The gradient side of the reference is
  `SLCEMonteCarlo._gradient_lane_ref!` by qualified name — an upstream rename
  breaks the a3 gate.
- **Checkpoint schema v2 ↔ `_RunSpec` compute fields ↔ both resume methods**
  (`checkpoint.jl`, `src/gpu/run.jl`): `run/compute`/`run/backend`/
  `run/workgroupsize` are trajectory-defining on the GPU path and validated on
  resume like `dt`/`seed`; the v1 back-read branch (compute = "cpu", ws = 0)
  stays alive as long as v1 files circulate; `_ck_due` must mirror
  `_ck_llg!`'s cadence exactly (the GPU loop snapshots only when it fires).
  Gates, one per clause: `test_gpu_llg.jl` "…resume refuses a compute switch" (CPU
  refuses a GPU file, GPU refuses a CPU file without `allow_compute_switch`, a changed
  `workgroupsize` is also a refused switch; mirrored for the quantum path in
  `test_checkpoint.jl`), the v1-rewrite resume in the same file (asserts
  `r.compute == "cpu"` and the ws `0 → 128` fallback), and the crash-shaped mid-run GPU
  resume for the cadence. That last one covers the dangerous direction only, by design: an
  under-firing `_ck_due` writes a stale host config and the bit-identity comparison
  diverges, while an over-firing one costs a needless `synchronize` and nothing else.
- **`equilibrium_stats` ↔ SLCEMonteCarlo's `_finalize_stats`**: `stats.jl`
  deliberately parallels the MC finalization (bin size
  `max(1, fld(kept, nbins))`, jackknife over `bin_means`, the
  `f(means, kT, n)` NamedTuple call, the `< 2` bins → NaN path) on the
  upstream **public** tier — if SLCEMonteCarlo changes its binning/jackknife
  conventions or `ObservableStat`, this bridge follows (gate: the cross-package
  equilibrium test compares stats produced by both pipelines). That includes
  `Evaluable`'s **`scope`**: `:spin` normalizes by `n_spin_active`, `:energy` by
  `n_active`, and `LLGResult` carries both counts so the bridge can honour it.
  It passed `n_active` unconditionally until 2026-07-28 — which is unreachable
  today (the two counts differ only when displacement-only sites exist, and a
  joint Hamiltonian dies at the first measurement — see the parked gap below)
  but was a trap laid for whatever lands spin–lattice dynamics here. Gate:
  `test_observables.jl`, on a directly-constructed divergent `LLGResult`.
- **S(q,ω) phase table ↔ upstream site ordering** (`sqw.jl` `_fill_phases!` /
  `channel_sumrule` ↔ SLCEMonteCarlo `site_index`/`site_atom`/
  `supercell_crystal`): the kernel assumes atom-fastest column-major cell order
  (`atom = mod1(s, n_a)`, `cell = (s−1) ÷ n_a` decomposed column-major) — the
  upstream documented, `test_geometry.jl`-pinned contract. Local gate: the
  ordering pin (kernel phases ≡ Cartesian `supercell_crystal` positions) and
  the translation-covariance test in `test_sqw_core.jl`.
- **S(q,ω) sign/normalization conventions ↔ the analytic gates**: the spatial
  sign (`_fill_phases!`'s single `cis(−…)`), the temporal kernel (forward FFT),
  the `1/√N` normalization, and the `Δt/(M·W₂)` spectral scale are frozen by
  `test_sqw_gates.jl` (Larmor closed forms, dimer, ring ±q asymmetry) and the
  Parseval/channel sum rules in `test_sqw_core.jl`. Change any one and the
  others (plus SPEC.md's estimator block) move together.
- **`_fft_pow2!` ↔ the reference DFT gate** (`fft.jl` ↔ `test_sqw_core.jl`):
  the own radix-2 kernel exists for determinism (no FFTW wisdom/threading
  variance); if FFTW ever replaces it behind the same seam, re-state the
  determinism scope in SPEC.md and keep the reference-DFT gate.
- **`trajectory_observable` ↔ the checkpoint series schema**: the recorded
  `:spins` series is persisted/validated by name and ncomp on resume —
  renaming the default or changing the `vec(to_matrix(…))` layout breaks
  stored checkpoint files (schema-version territory) and
  `structure_factor(path, …)`. Gates: name/ncomp by the path round-trip in
  `test_sqw_core.jl` and the resume validation in `test_checkpoint.jl`; the LAYOUT
  by the analytic ring gates (`test_sqw_gates.jl`, `test_quantum_thermostat.jl`
  G6) — verified by mutation, a transpose turns both red. The path round-trip
  alone cannot see it: writer and reader move together. `test_sqw_core.jl`
  "series layout is xyz-fastest-then-site" adds no coverage those gates lack; it
  states the on-disk contract against the configuration itself so a layout change
  fails at the source instead of surfacing as a wrong ring dispersion three files
  downstream.
- **`Observable` contract ↔ SLCEMonteCarlo's**: `run_llg` reuses
  `SLCEMonteCarlo.Observable` verbatim — `f(v::SLCEMonteCarlo.MCView)`, with
  `v.energy` = SLCE energy (intercept excluded, Zeeman NOT included) — so one
  definition measures identical values in both packages. `LLGResult.energies` is
  the *dynamical* energy (SLCE + Zeeman) — the two differ whenever `b_ext ≠ 0`
  (`test_observables.jl` pins the split). `_measure!` builds the view with an
  **empty** `disps`: spin dynamics has no displacement channel, so a displacement
  observable must throw here rather than read zeros. If SLCEMonteCarlo adds a
  field to `MCView`, decide explicitly what this package puts in it — silently
  passing an empty or zero placeholder is how a displacement observable would
  come to report a confident wrong number.
  **Known gap, deliberately unfixed (2026-07-27)**: `LLGProblem` accepts a joint
  (displacement-carrying) `TiledHamiltonian` without complaint, and the run then
  dies at the FIRST measurement with a `DimensionMismatch` from `MCView`'s inner
  constructor — loud, but pointing at the view rather than at the real problem.
  An upfront guard in `LLGProblem` is the fix; it is parked rather than applied
  because it is a scope statement ("spin dynamics does not do the displacement
  channel"), and that belongs with whatever decides how spin–lattice dynamics
  enters this package. Low severity precisely because it is loud: nothing here
  silently computes a wrong number on a joint model.
  Note also that the empty `disps` this package passes is inert: `MCView`'s
  constructor discards the argument on a pure-spin `H` anyway, so the enforcement
  lives entirely upstream (gated in SLCEMonteCarlo's `test_joint.jl`). Keep passing
  it regardless — it states the intent at the call site.
  **2026-07-29 (upstream M5-4)**: `MCView` gained an optional 5th field, the cell
  scale `strain` (`nothing` on a fixed cell; `SLCEMonteCarlo.strain(v)` throws
  then). This package's 4-arg construction is deliberately untouched — an LLG run
  is a fixed-cell run, so its views correctly carry `strain === nothing`, and a
  strain observable correctly throws here. If this package ever consumes a model
  under strain (a magnetoelastic LLG), the view must be built with the actual
  scale — the same explicit-decision rule as above. `ChainState` also gained a
  `strain` field and the MC checkpoint schema moved to v4; `model_fingerprint`'s
  mixing is UNCHANGED (pinned upstream, this package's checkpoint format depends
  on it).

## Tests

| Command | Purpose |
|---|---|
| `julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua (default) |
| `TEST_MODE=all julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua + JET |
| `TEST_MODE=jet julia --project -e 'using Pkg; Pkg.test()'` | JET type-stability |

Dev deps are path-devs: `Pkg.develop(path = "../SLCEMonteCarlo.jl")` and
`../SLCE.jl` (Manifest gitignored).

The GPU benches (`bench/bench_gpu_llg.jl`, `bench/bench_gpu_qtb.jl`) run in this
package's **own** environment, `bench/Project.toml` — it carries CUDA, which the
shared `@slce` env deliberately does not. Set it up once per machine:

```julia
Pkg.activate("bench")
Pkg.develop([PackageSpec(path = "../SLCE.jl"),
             PackageSpec(path = "../SLCEMonteCarlo.jl"), PackageSpec(path = ".")])
```

Its `Project.toml` is tracked and its `Manifest.toml` is not (the dev paths are
machine-specific). Do **not** borrow SLCEMonteCarlo's `bench/gpu` env for these:
that was the kugui arrangement until 2026-07-27, and it forced a permanently
divergent tracked file on the cluster — a downstream dep in the upstream repo,
invisible locally and rediscovered as a broken job.

## References

- `SPEC.md` — working equation, units, layout, validation pyramid, and the
  settled stochastic-LLG / Philox design (implement next).
- `STYLE_GUIDE.md` — package-specific style deltas.
- `references/` — supporting literature (notes tracked, PDFs local-only).
