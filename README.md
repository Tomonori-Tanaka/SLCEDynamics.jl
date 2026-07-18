# SCESpinDynamics.jl

Atomistic spin dynamics (Landau–Lifshitz–Gilbert) for fitted spin-cluster-expansion
(SCE) models — the dynamics member of the SCE family:

- [`SCEFitting.jl`](../SCEFitting.jl) fits the model (energy + torque co-fit),
- [`SCEMonteCarlo.jl`](../SCEMonteCarlo.jl) does equilibrium thermodynamics,
- **this package** integrates the real-time LLG equation of motion on the same
  tiled supercell Hamiltonian, through `SCEMonteCarlo`'s exact all-site gradient
  `energy_gradient!`.

## Scope

- LLG: `de_i/dt = p_i·[e_i×G_i + α_i e_i×(e_i×G_i)]`,
  `p_i = g_i/(ħ·magmom_i·(1+α_i²))`; energies in eV, time in fs, per-site
  `magmom` (μ_B) / `α` / `g`, uniform external field in tesla.
- Integrators: Depondt–Mertens (rotation-based Heun, norm-exact — default) and
  projected Heun (independent cross-check).
- **Stochastic LLG**: pass `temperature` [K] or `kT` [eV] to `run_llg` — keyed
  counter-based Philox thermal noise (stateless, Stratonovich), validated
  against analytic Boltzmann distributions and `SCEMonteCarlo` Metropolis
  equilibrium averages. `equilibrium_stats` computes τ_int-aware means and
  jackknifed `Evaluable`s (specific heat &c.) with the same definitions as the
  MC drivers.
- Bit-reproducible for any `ntasks`; deterministic runs consume no RNG (the
  `SCEMonteCarlo` P6 discipline).
- **Checkpoint/resume**: `run_llg(...; checkpoint = "run.jld2",
  checkpoint_interval = 50_000)` writes atomic plain-data JLD2 restart files;
  `resume(path, prob)` continues (or, via `nsteps`, *extends*) the run
  bit-identically to an uninterrupted one — the stateless noise means no RNG
  state is ever stored.
- **S(q,ω)**: record the trajectory with `trajectory_observable(H)`, then
  `structure_factor(res, H, crystal, q_path(crystal, verts; dims = H.dims).qs)`
  → the full Hermitian tensor on a two-sided ω axis (rad/fs + meV), elastic
  tensor separated, `sqw_plusminus`/`sqw_perp`/`sqw_trace` reductions, Welch
  averaging and seed ensembles. Conventions pinned by exact analytic gates
  (Larmor, dimer, ring dispersion, Parseval sum rules).

Planned next: GNEB, SIB, GPU. See `SPEC.md`.

## Quick start

```julia
using SCEFitting, SCEMonteCarlo, SCESpinDynamics

model = SCEFitting.load(SCEPredictor, "model.toml")
H = TiledHamiltonian(model; dims = (4, 4, 4))
prob = LLGProblem(H; magmom = 2.2, alpha = 0.1)          # magmom in μ_B
config0 = ...                                            # e.g. a ground state
res = run_llg(prob, config0; dt = 0.1, nsteps = 100_000, # dt in fs
              observables = standard_observables(H))     # MC observables plug in
res.times, res.energies, res.mean_spins                  # built-in time series
res.series[:absm]                                        # any Observable's series
```

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'               # unit + Aqua
TEST_MODE=all julia --project -e 'using Pkg; Pkg.test()' # + JET
```

The validation pyramid: analytic single-spin Larmor/damped-spiral, α = 0 energy
conservation at O(dt²), the analytic two-spin Heisenberg rotation, and bitwise
determinism gates.
