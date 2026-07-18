"""
Atomistic spin dynamics (Landau–Lifshitz–Gilbert) on fitted spin-cluster-expansion
models, on top of `SCEMonteCarlo.jl`'s tiled Hamiltonian and its exact all-site
gradient `energy_gradient!`.

The working equation, for unit spin directions `e_i` with per-site moment magnitude
`magmom_i` (μ_B units), gyromagnetic `g_i`, and Gilbert damping `α_i`:

    de_i/dt = g_i / (ħ · magmom_i · (1 + α_i²)) · [ e_i × G_i + α_i e_i × (e_i × G_i) ]

with `G_i = ∂E/∂e_i` (model energy units, eV for DFT-fitted models) the sum of the
SCE gradient and the Zeeman gradient `−magmom_i·μ_B·B_ext`. μ_B enters only at the
tesla boundary; the core evolution carries only `ħ` and the per-site `g/magmom`.
Time is in femtoseconds (`HBAR_EV_FS`). The torque convention matches the ecosystem
(`τ_i = −e_i × G_i = m_i × B_eff,i`, the physical / Landau–Lifshitz torque), and
the damping term strictly dissipates: `dE/dt = −Σ_i p_i α_i |G_⊥,i|² ≤ 0`.

Reproducibility follows the sibling packages' discipline: the deterministic
integrators consume no RNG and are bit-reproducible for any `ntasks`
(`SCEMonteCarlo`'s P6 scope — same package + Julia version).
"""
module SCESpinDynamics

using LinearAlgebra: norm, dot, cross
using StaticArrays
using SCEMonteCarlo: SCEMonteCarlo, TiledHamiltonian, n_sites, Observable
using SCEMonteCarlo: SpinConfig, site_atom, energy_gradient!

export HBAR_EV_FS, MU_B_EV_T
export LLGProblem, run_llg, LLGResult
export DepondtMertens, HeunProjected
# Re-exported from SCEMonteCarlo (the same binding — user observables written for
# the MC drivers plug into `run_llg` unchanged).
export Observable
public AbstractIntegrator

include("units.jl")
include("problem.jl")
include("integrators.jl")
include("run.jl")

end # module SCESpinDynamics
