# SLCEDynamics.jl

Atomistic spin dynamics for fitted SLCE (spin–lattice cluster expansion) models
from `SLCE.jl`: integrate the Landau–Lifshitz–Gilbert equation — deterministic,
stochastic (classical white noise), or semi-quantum (colored-noise thermostat) — on
`SLCEMonteCarlo.jl`'s tiled supercell Hamiltonian, with equilibrium statistics that
reuse the MC observable machinery, the dynamical structure factor ``S(q,\omega)``,
bit-reproducible checkpoint/restart, and a KernelAbstractions GPU path.

## Where it sits in the SLCE family

| Package | Role |
|---|---|
| `SLCE.jl` | fits the SLCE model (the input here) |
| `SLCEMonteCarlo.jl` | equilibrium thermodynamics: Metropolis, annealing, replica exchange |
| **`SLCEDynamics.jl`** | real-time dynamics: LLG / sLLG, ``S(q,\omega)``, QTB noise |

The fitted model enters only through `SLCEMonteCarlo`'s `TiledHamiltonian` and its
exact all-site gradient `energy_gradient!` — never through SLCE internals. Observable
and `Evaluable` definitions written for the MC drivers plug into the dynamics
drivers unchanged (the same bindings are re-exported here).

## Determinism and reproducibility

The package follows the sibling packages' discipline:

- **Deterministic runs consume no RNG.** The fixed-step integrators are RNG-free and
  bit-identical for any `ntasks` (the only threaded piece, `energy_gradient!`, is
  itself task-count independent).
- **Thermal noise is stateless.** Stochastic runs draw keyed counter-based Philox
  noise: a draw is a pure function of `(seed, site, step)`, so trajectories are
  bit-reproducible for any task count and checkpoints never store RNG state.
- **Checkpoints resume bit-identically** — the resumed (or *extended*) trajectory
  equals the uninterrupted run's exactly, tested with `==`, not `≈`.
- The scope is one package + Julia version (`SLCEMonteCarlo`'s P6 discipline); the
  GPU path adds (backend, workgroupsize) to its bitwise contract.

## Features

- LLG with per-site `magmom` (μ_B) / `α` / `g`, a uniform external field in tesla;
  energies in the model's units (eV), time in fs.
- Integrators: Depondt–Mertens (rotation-based Heun, norm-exact — the default) and
  projected Heun (an independent cross-check).
- Stochastic LLG validated against analytic Boltzmann distributions and
  `SLCEMonteCarlo` Metropolis equilibrium; [`equilibrium_stats`](@ref) reuses the MC
  binning/jackknife machinery.
- A semi-quantum colored-noise thermostat ([`QuantumThermostat`](@ref)) — magnon
  occupations follow Bose–Einstein statistics instead of classical equipartition.
- ``S(q,\omega)``: the full Hermitian tensor on a two-sided frequency axis, with the
  elastic tensor separated, Welch averaging, and seed ensembles — conventions pinned
  by exact analytic gates.
- Crash-safe JLD2 checkpoint/restart, and a GPU driver (`gpu_run_llg`) with the same
  noise stream and result semantics as the CPU path.

## Reading order

[Getting started](getting_started.md) → the guides
([dynamics](guide/dynamics.md), [thermal](guide/thermal.md),
[quantum thermostat](guide/quantum_thermostat.md),
[structure factor](guide/structure_factor.md),
[checkpointing](guide/checkpointing.md), [GPU](guide/gpu.md)) →
the [API reference](api.md).
Design decision records live in `docs/specs/` inside the repository.
