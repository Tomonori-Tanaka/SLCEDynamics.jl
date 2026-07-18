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
| `src/run.jl` | `run_llg` driver (fixed step; measurements at step 0, every `measure_interval`, and always at the final step) + `LLGResult`; user observables — the SAME `SCEMonteCarlo.Observable(name, ncomp, f)` definitions the MC drivers accept (`f(config, energy, H)`, fed the **SCE** energy so a definition measures identically in both packages) — recorded as `ncomp × n_measurements` time-series matrices in `LLGResult.series`; the shared stepping loop `_llg_loop!` (also the resume entry) |
| `src/checkpoint.jl` | JLD2 checkpoint/resume (schema below) — `run_llg(...; checkpoint, checkpoint_interval)` writer + `resume(path, prob::LLGProblem)` (a method of `SCEMonteCarlo.resume`, re-exported) |
| `src/fft.jl` | own power-of-two radix-2 FFT + Hann/rect windows (deliberately not FFTW — determinism; gated against a reference DFT) |
| `src/gpu/{state,kernels,run}.jl` | the GPU path: `GPULLGState`, the stage/noise/renorm kernels (literal `_omega`/`_rotate`/`_fill_noise!` ports), `run_llg_gpu`/`_llg_loop_gpu!`/the device `resume` method |
| `src/sqw.jl` | S(q,ω): `trajectory_observable`/`trajectory`, `q_path`, `structure_factor` (4 methods) → `SQWResult` (full 3×3 Hermitian tensor + separated elastic tensor), reductions `sqw_diag`/`sqw_trace`/`sqw_perp`/`sqw_plusminus`/`sqw_elastic`, axis helpers, `channel_sumrule` |

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

## Checkpoint / resume (implemented)

`run_llg(...; checkpoint = path, checkpoint_interval = n)` writes a plain-data
JLD2 file (schema v1, `kind = "llg"`; the sibling's format discipline — named
groups of Int/Float64/UInt64/String arrays, no Julia struct reconstruction,
atomic temp-file + `mv`) every `n` steps (`0` ⇒ completion only) and always at
completion. Because the noise is a stateless pure function of `(seed, site,
step)`, **no RNG state is stored**: the file carries the model fingerprint
(`SCEMonteCarlo.model_fingerprint`, the shared identity check), the
trajectory-defining parameters (problem arrays, `dt`, `nsteps`,
`measure_interval`, `renorm_interval`, integrator name, `kT`, `seed`), the
observable names/ncomps, the completed `step`, the bitwise configuration, and
the measurements recorded so far.

`resume(path, prob; observables, nsteps, …)` — a method of
`SCEMonteCarlo.resume` — validates everything, restores the configuration
**verbatim** (never through `from_matrix`, whose renormalization would perturb a
chaotic trajectory by ULPs), and continues the shared loop: bit-identical to the
uninterrupted run. A completed file reconstructs its `LLGResult` without
stepping (idempotent in retry loops). Since every per-step effect is keyed by
the absolute step index, `nsteps` may also *extend* the run — bit-identical to a
single longer run — refused only when a completed run's off-grid final
measurement would not be a prefix of the longer trace. Gates:
`test_checkpoint.jl` (crash-shaped mid-run files via a throwing observable,
extension ≡ uninterrupted, `==` throughout).

## S(q,ω) (implemented)

The classical dynamical structure factor from recorded trajectories, with every
sign and normalization frozen by exact deterministic gates (`test_sqw_gates.jl`,
`test_sqw_core.jl`). The estimator (fluctuation part, per active site):

    s^α(q,t)      = (1/√N) Σ_{s active} e^{−2πi f·x_s} (e_s^α(t) − ē_s^α)
    X^α(q,ω_k)    = Σ_{n<M} w_n s^α(q,t_n) e^{−2πikn/M}
    S^{αβ}(q,ω_k) = (Δt_s/(M·W₂)) · X^α(q,ω_k)^* X^β(q,ω_k)
    S_el^{αβ}(q)  = m̄^α(q)^* m̄^β(q),   m̄ = (1/√N) Σ_s e^{−2πi f·x_s} ē_s

- **Acquisition**: the trajectory is an ordinary observable
  (`trajectory_observable(H)`, `ncomp = 3n`, `vec(to_matrix(config))`) — it
  inherits the measurement cadence, checkpoint persistence (a completed
  checkpoint file IS the trajectory file format; `structure_factor(path, …)`
  reads it), resume/extension, and bit-reproducibility with zero new machinery.
  In-RAM cost `24·n·n_meas` bytes — keep ≲ 10 GB (a chunked writer and a
  q-projected recorder are deferred seams).
- **q**: fractional coordinates in the training-cell reciprocal lattice
  (`q_cart = 2π·Bᵀ·f`; SCEFitting's `reciprocal` carries no 2π — it enters here
  exactly once). Only supercell-commensurate q (`f_i·N_i ∈ ℤ`) are
  representable: `structure_factor` throws on others; `q_path(…; dims)` snaps
  loudly (`qs` vs `qs_requested`). Phases need no Cartesian positions —
  `q·r_s = 2π f·(cell + frac_atom)` identically, with the cell part in exact
  integer arithmetic.
- **Mean/elastic**: the per-site time mean over the analysis window (one global
  mean, never per segment) is always subtracted; the elastic tensor is reported
  separately (`S_el`) so Bragg weight cannot leak through the window into the
  inelastic spectrum.
- **Axes/components**: two-sided fftshifted ω [rad/fs] (for even `nfft` the
  Nyquist bin exists only at −M/2) + `energies_mev`; the full 3×3 Hermitian
  tensor is stored (all reductions are cheap post-contractions): `sqw_diag`,
  `sqw_trace`, `sqw_perp` (NaN at Γ), `sqw_plusminus` (the ω-sign-resolving
  transverse channel; positive precession about `+axis` → +ω).
- **Signs** (gate-pinned, one code site each): spatial `e^{−2πi f·x}` — pinned
  by the ring gate's +q/−q asymmetry and the translation-covariance gate;
  temporal `e^{−iωt}` — pinned by the Larmor closed-form (all `S^{+−}` weight
  at +ω_L).
- **Statistics**: Welch (`nsegments`, `overlap`, `:hann`/`:none`, derived
  power-of-two `nfft`, trailing samples truncated), `discard` thermalization
  prefix, seed-ensemble averaging (`Vector{LLGResult}`; spectra averaged, never
  amplitudes) with realization standard errors at R ≥ 3. Caveat (documented):
  Hann + mean subtraction still leaves a residual low-ω leak on thermal data.
- **Exact gates**: per-(q,α) Parseval (`Σ_k S/(MΔt) ≡` windowed time average,
  any window); the global sum rule at the sublattice-channel level
  (`channel_sumrule`: `Σ_{q,a,α,k} S_aa/(MΔt) = Σ_active (1−|ē|²)`; the
  unfolded S is deliberately NOT one-BZ periodic); Larmor single-bin closed
  forms (`S^{xx} = MΔt/4` at ±ω_L, bin sum 1/2, `S^{+−} = MΔt` at +ω_L only);
  the dimer's conserved-q=0 null + single mode at q = (0,0,2) with
  `Ω = 2p|J_pair|cosθ` (an exact rigid-rotation solution, J measured from
  tiled energies); the 4-site ring's exact spiral dispersion
  `ω = 2p|J_eff|cosθ(1−cos(2πm/4))` with the (+q, +ω) placement and magmom
  scaling. `structure_factor` is a pure function of its inputs — bit-identical
  across calls and for any `ntasks`.
- **Deferred**: q-projected/streaming recording, form factors and
  magmom-weighted intensities, stored chiral channels, full-grid FFT(W) mode,
  the classical→quantum `βħω/(1−e^{−βħω})` intensity factor, powder averaging,
  local-frame transverse splits, GPU.

## GPU path (implemented; A100 go/no-go pending)

`run_llg_gpu(prob, config0, gH; …)` (public unexported until the go/no-go;
decision record `docs/specs/gpu-llg.md` + SCEMonteCarlo's G7) runs both
integrators, deterministic and sLLG, on a KernelAbstractions backend over
SCEMonteCarlo's device gradient (`gpu_energy_gradient!`). Key properties:

- The noise stream is the SAME stateless Philox `(seed, site, step)` as the CPU
  path — a same-seed CPU and GPU run are one stochastic realization, differing
  only through gradient-fold/rotation roundoff. Measurements/checkpoints run on
  host snapshots (downloaded only on measurement/checkpoint events), so
  `LLGResult`/`equilibrium_stats`/`structure_factor`/checkpoint files are
  identical in kind to the CPU path's.
- Determinism: bitwise for fixed (`seed`, backend, `workgroupsize`, package +
  Julia version). The bitwise CI tier compares against the composite keyed
  reference (upstream `_gradient_lane_ref!` + literal stage expressions) on the
  KA-CPU backend; CPU↔GPU is a tolerance/statistical comparison only.
- Checkpoint schema v2 records (`compute`, `backend`, `workgroupsize`); v1
  back-read. `resume(path, prob, gH)` continues/extends bit-identically on the
  same backend+ws; any compute switch needs `allow_compute_switch = true`
  (same realization, not bit-identical — loud by design).
- Gates (`test_gpu_llg.jl`, all local): noise kernel ≡ host `_fill_noise!`
  bitwise (incl. the inactive exact-+0.0 rule); full driver ≡ composite keyed
  reference bitwise (both integrators, det + sLLG, ws ∈ {4, 32});
  repeat/seed/ws sensitivity; GPU checkpoint/resume/extension bitwise; schema
  v1 back-read; device Larmor exactness; CPU-vs-GPU tolerance on the
  non-chaotic dimer; FDT pipeline smoke. **No performance claims until the
  A100 bench (`bench/bench_gpu_llg.jl`) measures the l044 8³ step — the ≥ 5×
  go/no-go bar.**

## Planned

- Seams kept open: integrator dispatch (`_step!`), additive torque terms beyond
  energy gradients (STT/SOT), observable callback at stride, `NoiseModel`,
  device-side observable reductions / async-overlap measurement.
- Later: GNEB, Mentink SIB, adaptive dt, GPU S(q,ω), multi-GPU.
