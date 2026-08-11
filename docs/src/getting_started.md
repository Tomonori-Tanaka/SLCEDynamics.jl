# Getting started

```@meta
CurrentModule = SLCEDynamics
```

## Install

The package lives alongside its model source `SLCE.jl` and its Hamiltonian /
gradient layer `SLCEMonteCarlo.jl`; during development all three are path-devs:

```julia
using Pkg
Pkg.develop(path = "path/to/SLCE.jl")
Pkg.develop(path = "path/to/SLCEMonteCarlo.jl")
Pkg.develop(path = "path/to/SLCEDynamics.jl")
```

## From a fitted model to a trajectory

In production the model comes from a fitted file
(`SLCE.load(SLCEModel, "model.toml")`). Here we build a small reference
model through the same fitted-model surface — the classical Heisenberg ferromagnet
on a simple cubic lattice, one coefficient (`< 0` ⇒ ferromagnetic):

```@example gs
using SLCEDynamics, SLCEMonteCarlo, SLCE
import Spglib                      # activates SLCE's SpglibBackend extension
using LinearAlgebra

lat = Lattice(Matrix(1.0 * I(3)))
cell = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
spec = BasisSpec(; nbody = 2, cutoff = 1.1, lmax = [1], soc = false)
basis = SLCEBasis(cell, spec; backend = SpglibBackend(), images = AllImages())
model = SLCEModel(basis, 0.0, [-0.01])

H = TiledHamiltonian(model; dims = (2, 2, 2))       # training cell → 2×2×2 supercell
prob = LLGProblem(H; magmom = 2.2, alpha = 0.5)     # magmom in μ_B
```

[`LLGProblem`](@ref) resolves the LLG prefactor and the (here zero) Zeeman gradient
once at construction; `magmom`, `alpha`, and `g` accept a scalar, a per-training-cell
-atom vector, or a per-site vector.

## A deterministic ringdown

Start from a random configuration and let the damping pull it into the ferromagnetic
ground state — a deterministic run (no temperature) consumes no RNG at all:

```@example gs
using Random

rng = Xoshiro(1)
# random unit spins — normalized deliberately: `from_matrix` validates
# directions and refuses a scaled column rather than normalizing it silently
m0 = randn(rng, 3, n_sites(H))
config0 = SLCEMonteCarlo.from_matrix(m0 ./ sqrt.(sum(abs2, m0; dims = 1)))

res = run_llg(prob, config0; dt = 0.5, nsteps = 600,            # dt in fs
              measure_interval = 10,
              observables = standard_observables(H))            # MC observables plug in
```

## Inspecting the result

[`LLGResult`](@ref) carries the built-in time series (`times` [fs], `energies` —
SLCE + Zeeman — and the mean spin direction over active sites), one series matrix per
requested observable, and the final configuration. The damping strictly dissipates:
the energy falls monotonically toward the aligned state, and the mean-spin length
grows toward 1:

```@example gs
(E_start = res.energies[1], E_end = res.energies[end],
 m_start = norm(res.mean_spins[1]), m_end = norm(res.mean_spins[end]))
```

Any observable's series is an `ncomp × n_measurements` matrix aligned with
`res.times`:

```@example gs
res.series[:absm][:, end-3:end]        # |m| over the last four measurements
```

## Where to go next

- Deterministic LLG in detail — integrators, units, `dt` guidance, energy
  conservation at ``\alpha = 0``: [dynamics](guide/dynamics.md).
- Finite temperature — the stochastic LLG, seeding, and
  [`equilibrium_stats`](@ref): [thermal](guide/thermal.md); the semi-quantum
  colored-noise thermostat: [quantum thermostat](guide/quantum_thermostat.md).
- Spectra — record a trajectory and compute ``S(q,\omega)``:
  [structure factor](guide/structure_factor.md).
- Long runs — crash-safe restart files and bit-identical resume/extension:
  [checkpointing](guide/checkpointing.md); the device driver: [GPU](guide/gpu.md).
