# Stochastic LLG (finite temperature)

```@meta
CurrentModule = SLCEDynamics
```

Passing a temperature to [`run_llg`](@ref) switches on the thermal field: per active
site and step, a Gaussian gradient contribution ``\mathbf G_{\mathrm{th},i} =
\sigma_i\,\boldsymbol\xi_i`` with

```math
\sigma_i = \sqrt{\frac{2\,\alpha_i\, k_B T\, \hbar\, \mathrm{magmom}_i}{g_i\,\Delta t}}
\quad [\mathrm{eV}],
\qquad \boldsymbol\xi_i \sim N(0, I_3),
```

the fluctuation–dissipation amplitude for this parametrization (no ``(1+\alpha^2)``
factor — the noise rides the same prefactored equation as the deterministic field).
The same draw feeds **both** Heun stages of one step; that is what makes the scheme
converge to the Stratonovich solution. Noise requires ``\alpha > 0`` on every active
site — an undamped spin never thermalizes.

## Temperature convention

Exactly one of two keywords, the ecosystem-wide rule:

- `temperature` — kelvin, converted internally with `SLCEMonteCarlo.KB_EV` (assumes
  an eV-fitted model, the convention for DFT-fitted models);
- `kT` — ``k_B T`` directly in the model's energy units (theory / test runs).

`run_llg` takes a single scalar (no ladders — one trajectory per run).

## Seeding and the stateless noise stream

Draws are keyed counter-based Philox (`SLCEMonteCarlo`'s Random123-gated facade): a
draw is a **pure function of `(seed, site, step)`**. Consequences worth relying on:

- The trajectory is a pure function of
  (`prob`, `config0`, `dt`, `nsteps`, `integrator`, `seed`) — no RNG state exists,
  and results are **bit-identical for any `ntasks`**.
- Checkpoints never store RNG state (see [checkpointing](checkpointing.md)).
- By default every thermostatted run draws a fresh `seed = rand(UInt64)`, recorded
  in `LLGResult.seed`, so repeated runs are independent realizations; pass an
  explicit `seed` for bit-reproducibility (tests, docs, seed ensembles).

```@example therm
using SLCEDynamics, SLCEMonteCarlo, SLCE
import Spglib
using LinearAlgebra, Random

lat = Lattice(Matrix(1.0 * I(3)))
cell = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
basis = SLCEBasis(cell, BasisSpec(; nbody = 2, cutoff = 1.1, lmax = [1],
                                 isotropy = true);
                 backend = SpglibBackend(), images = AllImages())
model = SLCEModel(basis, 0.0, [-0.01])
H = TiledHamiltonian(model; dims = (2, 2, 2))
prob = LLGProblem(H; magmom = 2.2, alpha = 0.5)
config0 = SLCEMonteCarlo.from_matrix(randn(Xoshiro(1), 3, n_sites(H)))

res = run_llg(prob, config0; dt = 0.5, nsteps = 4000, kT = 0.01, seed = 42,
              observables = standard_observables(H))
res2 = run_llg(prob, config0; dt = 0.5, nsteps = 4000, kT = 0.01, seed = 42,
               ntasks = 2, observables = standard_observables(H))
res.energies == res2.energies          # bitwise, independent of the task count
```

## Equilibrium statistics

[`equilibrium_stats`](@ref) bridges the recorded time series to `SLCEMonteCarlo`'s
binning/jackknife machinery, so the **same** `Evaluable` definitions the MC drivers
accept (specific heat, susceptibility, Binder, user-defined) work on sLLG data:

```@example therm
stats = equilibrium_stats(res)         # discards the first half by default
(energy = stats[:energy].mean[1], err = stats[:energy].err[1],
 tau_int = stats[:energy].tau_int[1],
 specific_heat = stats[:specific_heat].mean[1])
```

- Raw observables get autocorrelation-aware means/errors (`ObservableStat`, via
  `LogBinner`); evaluables are jackknifed over equal-weight bins of their scalar
  inputs, with `f(means, kT, n)` receiving the run's `kT` and active-site count.
- `discard` drops the leading measurements as thermalization (default: the first
  half). The sLLG equilibration time scales like ``1/\alpha`` — inspect the energy
  series when in doubt.
- The default evaluables need `:energy`, `:energy2`, `:m2`, `:m4`, `:absm` among the
  recorded observables — record `SLCEMonteCarlo.standard_observables(H)` in the run.

!!! warning "O(dt) equilibrium bias"
    The Heun-family sLLG has weak order 1, so equilibrium averages carry an
    ``O(dt)`` bias — check `dt`-sensitivity before trusting a tight comparison.

## Cross-validation against Metropolis

The sLLG equilibrium is gated in the test suite against `SLCEMonteCarlo.run_mc`
Metropolis averages (``\langle E\rangle``, spin correlators, at 3σ with
τ_int-aware errors on both sides), and the noise parametrization is pinned by an
α-independence Boltzmann gate — a wrong ``(1+\alpha^2)`` would be a
``(1+\alpha^2)\times`` temperature error. The same comparison, live:

```@example therm
mc = run_mc(H; kT = 0.01, sweeps_therm = 2000, sweeps_measure = 8000, seed = 7)
(sllg = stats[:energy].mean[1], sllg_err = stats[:energy].err[1],
 metropolis = mc.points[1].stats[:energy].mean[1],
 metropolis_err = mc.points[1].stats[:energy].err[1])
```

Two independent samplers of the same Boltzmann ensemble — the means agree within
their (τ_int-aware) error bars. This classical cross-check applies to the default
[`ClassicalThermostat`](@ref) only; a [`QuantumThermostat`](@ref) run equilibrates
to a *different* (non-Boltzmann) ensemble — see the
[quantum thermostat](quantum_thermostat.md) guide.
