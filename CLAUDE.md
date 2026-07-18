# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

## Project goal

Atomistic spin dynamics (LLG / stochastic LLG) on fitted SCE models, on top of
`SCEMonteCarlo.jl` (`TiledHamiltonian` + `energy_gradient!`). Priority: physical
correctness of the equation of motion and thermostat (validated against analytic
solutions and, for finite T, against `SCEMonteCarlo` equilibrium averages) and
the ecosystem's bit-reproducibility discipline. See `SPEC.md` for the working
equation, unit system, and the settled stochastic-LLG design.

## Numerical / physics conventions

- **Units**: energy in the model's units (eV), time in **fs** (`HBAR_EV_FS`),
  external field in **tesla** only at the boundary (`MU_B_EV_T`); μ_B cancels in
  the core evolution (`p_i = g_i/(ħ·magmom_i·(1+α_i²))` carries only ħ and
  g/magmom). Temperature (when the thermostat lands): reuse
  `SCEMonteCarlo.resolve_kt` — `temperature`[K] XOR `kT`[eV], never both.
- **Spins are unit vectors** (`SpinConfig = Vector{SVector{3,Float64}}`);
  `magmom` [μ_B] is a separate per-site parameter (an explicit `LLGProblem`
  argument — a fitted `SCEPredictor` does not carry per-atom moments).
- **Sign conventions**: torque `τ_i = −e_i×G_i = m_i×B_eff,i` (physical /
  Landau–Lifshitz, the ecosystem convention); rotation vector
  `ω = −p(G + α e×G)`; dissipation `dE/dt = −Σ p α|G_⊥|² ≤ 0`. γ > 0; a spin at
  +x in B ∥ +z moves toward +y. Larmor anchor: g = 2, 1 T → 27.9925 GHz.
- **Zeeman gradient is NOT tangent-projected** (deliberate — makes constant-field
  Depondt rotation exact); the SCE gradient arrives projected from
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
- **Inactive-site convention ↔ `SCEMonteCarlo`'s frozen-spin convention**:
  integrator stages skip, no Zeeman, `mean_spins` excludes, `pref = 0`, spins
  bitwise frozen (`test_single_spin.jl` frozen testset). Mirrors the sibling —
  if `SCEMonteCarlo` changes `site_active` semantics, this package follows.
- **Upstream gradient contract**: `energy_gradient!` (tangent-projected, exact,
  bit-identical for any ntasks, ≈ 1 sweep/call) is pinned in SCEMonteCarlo's
  `test_gradient.jl`; this package's determinism gates assume it.
- **FDT constant ↔ `_omega`'s (1+α²) prefactor ↔ the α-independence gate**:
  `σ_i = √(2 α kB T ħ magmom/(g Δt))` in `noise.jl` is correct ONLY for noise
  entering the (1+α²)-prefactored equation that `_omega` implements — change
  either parametrization and the other (and design-notes in SPEC.md) must move;
  the runtime tripwire is the α = 0.5 vs 1.0 Boltzmann gate in
  `test_thermostat.jl`. The same draw must feed BOTH integrator stages
  (Stratonovich) — `_fill_noise!` runs once per step, never per stage.
- **Noise counter layout ↔ SCEMonteCarlo's philox contract**: word 4 carries the
  nonzero `_DOMAIN_SD` tag (MC streams use 0 — the documented upstream
  contract); `philox_block`/`philox_normal2` are the public facade pinned by the
  Random123 known-answer test upstream. Change the layout and seeded
  trajectories change (breaking-note territory, the P6 scope).
- **Checkpoint writer ↔ reader ↔ the absolute-step purity of `_llg_loop!`**
  (`checkpoint.jl`, `run.jl`): resume bit-identity rests on every per-step
  effect being a pure function of the absolute step index — the Philox counter
  (`_noise_ctrs(site, step)`), the renorm cadence (`step % renorm_interval`),
  and the measurement grid (`step % mi`, plus the final step). Introduce ANY
  carried-over per-step state (an accumulated time, a cached field, a stage
  seed) and the checkpoint schema must grow to carry it (schema-version bump).
  The configuration is restored **verbatim** (`_config_verbatim`) — never
  `from_matrix`, whose renormalization ULP-perturbs chaotic trajectories (the
  bug the crash-resume gate in `test_checkpoint.jl` caught). The model identity
  is `SCEMonteCarlo.model_fingerprint` (upstream public facade — its mixing is
  part of this file format too).
- **`equilibrium_stats` ↔ SCEMonteCarlo's `_finalize_stats`**: `stats.jl`
  deliberately parallels the MC finalization (bin size
  `max(1, fld(kept, nbins))`, jackknife over `bin_means`, the
  `f(means, kT, n_active)` NamedTuple call, the `< 2` bins → NaN path) on the
  upstream **public** tier — if SCEMonteCarlo changes its binning/jackknife
  conventions or `ObservableStat`, this bridge follows (gate: the cross-package
  equilibrium test compares stats produced by both pipelines).
- **`Observable` contract ↔ SCEMonteCarlo's**: `run_llg` reuses
  `SCEMonteCarlo.Observable` verbatim (`f(config, energy, H)`, `energy` = SCE
  energy, intercept excluded, Zeeman NOT included) so one definition measures
  identical values in both packages. `LLGResult.energies` is the *dynamical*
  energy (SCE + Zeeman) — the two differ whenever `b_ext ≠ 0`
  (`test_observables.jl` pins the split). If SCEMonteCarlo changes the
  `Observable` measurement signature, `_measure!` follows.

## Tests

| Command | Purpose |
|---|---|
| `julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua (default) |
| `TEST_MODE=all julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua + JET |
| `TEST_MODE=jet julia --project -e 'using Pkg; Pkg.test()'` | JET type-stability |

Dev deps are path-devs: `Pkg.develop(path = "../SCEMonteCarlo.jl")` and
`../SCEFitting.jl` (Manifest gitignored).

## References

- `SPEC.md` — working equation, units, layout, validation pyramid, and the
  settled stochastic-LLG / Philox design (implement next).
- `STYLE_GUIDE.md` — package-specific style deltas.
- `references/` — supporting literature (notes tracked, PDFs local-only).
