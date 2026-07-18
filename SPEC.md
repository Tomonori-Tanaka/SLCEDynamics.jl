# SCESpinDynamics.jl — architecture

Atomistic spin dynamics (LLG / stochastic LLG) on fitted SCE models. Consumes
`SCEMonteCarlo`'s `TiledHamiltonian` and its public all-site gradient
`energy_gradient!` — never SCE internals.

## Physics

Working equation for unit spins `e_i`, moment magnitudes `magmom_i` [μ_B],
gyromagnetic `g_i`, Gilbert damping `α_i`:

    de_i/dt = p_i · [e_i × G_i + α_i e_i × (e_i × G_i)]
    p_i     = g_i / (ħ · magmom_i · (1 + α_i²))          [1/(eV·fs)]
    G_i     = ∂E/∂e_i = G_SCE,i + G_Zeeman,i             [eV]

Derived from the standard LLG `de/dt = −γ/(1+α²)[e×B + α e×(e×B)]` with
`B_i = −G_i/(magmom_i·μ_B)`, `γ_i = g_i μ_B/ħ` — μ_B cancels; it survives only in
the Zeeman gradient `G_Zeeman,i = −magmom_i·μ_B·B_ext` (field in tesla). Torque
convention matches the ecosystem: `τ_i = −e_i×G_i = m_i×B_eff,i` (physical /
Landau–Lifshitz). Dissipation identity: `dE/dt = −Σ_i p_i α_i |G_⊥,i|² ≤ 0`,
exactly zero at `α = 0` — the conservation gate. Rotation-vector form (what the
integrators step): `de/dt = ω×e`, `ω_i = −p_i(G_i + α_i e_i×G_i)`.

Units: energy = model units (eV), time = fs (`HBAR_EV_FS`), field = tesla at the
boundary (`MU_B_EV_T`). Anchor: g = 2, 1 T → 27.9925 GHz (`test_units.jl`).

## Module layout

| File | Contents |
|---|---|
| `src/units.jl` | `HBAR_EV_FS` (exact SI ratio), `MU_B_EV_T` (CODATA-2018). `KB_EV`/`resolve_kt` are reused from `SCEMonteCarlo`, never redefined |
| `src/problem.jl` | `LLGProblem` (Hamiltonian + per-site `magmom`/`alpha`/`g` + `b_ext`; prefactor `pref` and Zeeman gradient `gzee` resolved at construction; scalar / per-atom / per-site parameter resolution via `site_atom`), `SCEMonteCarlo.total_energy(::LLGProblem, config)` = SCE + Zeeman |
| `src/integrators.jl` | `DepondtMertens` (rotation Heun, norm-exact, default), `HeunProjected` (independent cross-check); `_omega` / `_rotate` / `_step!` |
| `src/run.jl` | `run_llg` driver (fixed step; measurements at step 0, every `measure_interval`, and always at the final step) + `LLGResult`; user observables — the SAME `SCEMonteCarlo.Observable(name, ncomp, f)` definitions the MC drivers accept (`f(config, energy, H)`, fed the **SCE** energy so a definition measures identically in both packages) — recorded as `ncomp × n_measurements` time-series matrices in `LLGResult.series` |

## Conventions and invariants

- **Inactive sites** (`H.site_active`): frozen bitwise — skipped by both
  integrator stages, no Zeeman, excluded from `mean_spins`, `pref = 0`. Mirrors
  `SCEMonteCarlo`'s frozen-spin convention. `magmom` is validated (> 0, finite)
  on active sites only.
- **Gradient contract**: `energy_gradient!` returns the tangent-projected
  `G_SCE` (`e·G = 0` exact). The constant Zeeman gradient is deliberately NOT
  projected: the radial part of `ω` is null in `ω×e`, and keeping it makes the
  Depondt rotation axis exactly `B_ext` for a pure field — uniform Larmor
  precession is then exact per step at any `dt`.
- **Determinism**: the deterministic integrators consume no RNG; trajectories
  are bit-identical for any `ntasks` (inherited from `energy_gradient!`'s
  task-count independence + serial per-site update loops). Same P6 scope as
  `SCEMonteCarlo` (same package + Julia version).
- **Time** is `step × dt`, never an accumulated float.
- Cost: one step = 2 field evaluations ≈ 2 Metropolis sweeps.

## Validation pyramid (test/unit/)

(a) `test_single_spin.jl` — analytic Larmor (Depondt exact for constant field;
    magmom-independence of the Zeeman frequency) + damped log-tan spiral with
    observed order 2; (b) `test_conservation.jl` — α = 0 drift O(dt²) on an
    anisotropic l ≤ 2 model, norm preservation to 1e-12 without renormalization,
    strict dissipation, Depondt↔Heun cross-agreement, bitwise/ntasks
    determinism; (c) `test_two_spin.jl` — analytic uniform rotation about the
    conserved total spin (`ω = −pJs`), magmom halves the frequency, B_ext adds
    the Larmor rate, damped relaxation to the ferro ground state.

## Stochastic LLG (implemented)

`run_llg(...; temperature/kT, seed)` switches on the thermal field: per active
site and step, `G_th,i = σ_i·ξ_i` with `σ_i = √(2 α_i kB T ħ magmom_i/(g_i Δt))`
[eV], `ξ ~ N(0, I₃)` — the eV-side form of the fluctuation–dissipation constant
`D = α·kB·T/(γ·magmom·μ_B)`, which carries **no `(1+α²)`** in this
parametrization (it maps to García-Palacios–Lázaro under `γ_L = γ/(1+α²)`,
`λ = α`; `src/noise.jl`). The SAME draw feeds both Heun stages (Stratonovich).
Requires `α > 0` on every active site. Weak order 1 ⇒ O(dt) equilibrium bias.

Noise draws are keyed philox4x32-10 through `SCEMonteCarlo.philox_block` /
`philox_normal2` (public facade, Random123 known-answer-gated): counter
`(site, step_lo32, slot ∈ {0,1}, 0x5344_0000 | step_hi16)` — the nonzero "SD"
word-4 tag keeps every stream disjoint from MC's GPU streams under a shared
seed; step capacity 2^48; four normals per (site, step), the fourth discarded
(reserved). A draw is a pure function of (seed, site, step): no RNG state, and
trajectories stay bit-identical for any `ntasks`. Default `seed = rand(UInt64)`
(the sibling convention), recorded in `LLGResult.seed`.

`equilibrium_stats(res; evaluables, discard, nbins)` (`src/stats.jl`) bridges
the recorded series to `SCEMonteCarlo`'s public binning machinery
(`LogBinner`/`BinStore`/`bin_means`/`jackknife`): raw observables get
τ_int-aware `ObservableStat`s, and the SAME `Evaluable` definitions the MC
drivers accept (specific heat, susceptibility, Binder, user-defined) are
jackknifed with `f(means, res.kT, res.n_active)`. Thermostatted runs only.

Gates (`test_thermostat.jl`): single-spin uniaxial Boltzmann vs analytic
quadrature at α = 1.0 AND α = 0.5 (α-independence — a wrong `(1+α²)` is a
(1+α²)× temperature error, measured agreement ~0.1σ), noise determinism
(seed/ntasks/temperature-route), sphere preservation under noise, and the
cross-package gate: sLLG ≡ `run_mc` Metropolis equilibrium (⟨E⟩, ⟨e₁·e₂⟩ at 3σ
with τ_int-aware errors on both sides) on the dimer.

## Planned

- Seams kept open: integrator dispatch (`_step!`), additive torque terms beyond
  energy gradients (STT/SOT), observable callback at stride, `NoiseModel`.
- JLD2 checkpoint/resume (stateless noise ⇒ carries only seed + step + config).
- Later: S(q,ω) from trajectory dumps, GNEB, Mentink SIB, adaptive dt, GPU
  (needs the device ∇Z in SCEMonteCarlo — its phase 2).
