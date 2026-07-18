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

using JLD2: jldopen
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
using LinearAlgebra: norm, normalize, dot, cross
using LinearAlgebra: I, kron, eigen, Symmetric, Diagonal
using StaticArrays
using Statistics: mean
using SCEFitting: Crystal, n_atoms
using SCEMonteCarlo: SCEMonteCarlo, TiledHamiltonian, n_sites, Observable
using SCEMonteCarlo: SpinConfig, site_atom, energy_gradient!
using SCEMonteCarlo: resolve_kt, philox_block, philox_normal2
using SCEMonteCarlo: Evaluable, ObservableStat, standard_evaluables
using SCEMonteCarlo: LogBinner, BinStore, bin_means, jackknife, std_error, tau_int
using SCEMonteCarlo: model_fingerprint, resume

export HBAR_EV_FS, MU_B_EV_T
export LLGProblem, run_llg, LLGResult
export DepondtMertens, HeunProjected
export equilibrium_stats
export structure_factor, SQWResult, q_path, trajectory_observable, trajectory
export sqw_diag, sqw_trace, sqw_perp, sqw_plusminus, sqw_elastic
export homega_ev, homega_mev, freq_thz
# Re-exported from SCEMonteCarlo (the same bindings — observable/evaluable
# definitions written for the MC drivers plug into this package unchanged, and
# `resume` gains an `LLGProblem` method here).
export Observable, Evaluable, resume
public AbstractIntegrator
public channel_sumrule
# GPU path (public-unexported until the A100 go/no-go, mirroring the sibling)
public run_llg_gpu, GPULLGState
# Quantum thermostat (public-unexported until the pinned Barker–Bauer fit
# constants land — the current filter is the identity wiring placeholder)
public ClassicalThermostat, QuantumThermostat, AbstractThermostat
public ColoredNoiseFilter

include("units.jl")
include("problem.jl")
include("integrators.jl")
include("noise.jl")
include("thermostat.jl")
include("run.jl")
include("checkpoint.jl")
include("stats.jl")
include("fft.jl")
include("sqw.jl")
include("gpu/state.jl")
include("gpu/kernels.jl")
include("gpu/run.jl")

end # module SCESpinDynamics
