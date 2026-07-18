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

## Planned (design settled, not yet implemented)

- **Stochastic LLG**: noise field added to `B` inside the working equation;
  `D = α·kB·T/(γ·magmom·μ_B)` — no `(1+α²)` in this parametrization (maps to
  García-Palacios–Lázaro under `γ_L = γ/(1+α²)`, `λ = α`). Implementation form
  `G_th,i = σ_i·ξ_i` per step, `σ_i = √(2 α_i kB T ħ magmom_i/(g_i Δt))` [eV],
  the SAME draw shared by both Heun stages (Stratonovich). Keyed Philox4x32-10,
  counter `(site, step, slot, domain-tag)` — stateless, checkpoint carries only
  `(seed, n_step, dt)`; widen/split the step word before 2³¹ steps. Gates:
  single-spin Boltzmann histogram α-independent between α = 0.1 and 1.0 (a wrong
  `(1+α²)` is a 2× temperature error at α = 1), then ⟨E⟩/⟨|m|⟩ vs
  `SCEMonteCarlo` Metropolis at matched T (3σ, τ_int-aware binning both sides,
  dt-halving column).
- Seams kept open: integrator dispatch (`_step!`), additive torque terms beyond
  energy gradients (STT/SOT), observable callback at stride, `NoiseModel`.
- Later: S(q,ω) from trajectory dumps, GNEB, Mentink SIB, adaptive dt, GPU
  (needs the device ∇Z in SCEMonteCarlo — its phase 2).
