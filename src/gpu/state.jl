# Device-resident LLG state (decision record docs/specs/gpu-llg.md). SD owns
# its device arrays (`SCEMonteCarlo`'s `GPUChainState` carries MC bookkeeping
# dynamics has no use for); the upstream contract is the shared
# `GPUTiledHamiltonian`, the stateless `gpu_energy_gradient!` entry point with
# its `GPUGradientScratch`, and plain `copyto!` transfers.

"""
    GPULLGState(gH::GPUTiledHamiltonian, prob::LLGProblem, config0::SpinConfig,
                sigma::Vector{Float64}) -> GPULLGState

Device arrays for one [`run_llg_gpu`](@ref): the configuration and the
integrator scratch (`depred`/`domega1`/`dG`/`dgth` — the device sibling of
`_LLGScratch`), the per-site problem parameters uploaded once
(`dgzee`/`dpref`/`dalpha`/`dsigma`/`dactive`), the upstream gradient scratch,
and a preallocated host snapshot buffer for measurement/checkpoint downloads.
"""
struct GPULLGState{VC<:AbstractVector{SVector{3,Float64}},
                   VF<:AbstractVector{Float64},VB<:AbstractVector{Int8},
                   GS<:SCEMonteCarlo.GPUGradientScratch}
    dconfig::VC
    depred::VC
    domega1::VC
    dG::VC
    dgth::VC
    dgzee::VC
    dpref::VF
    dalpha::VF
    dsigma::VF
    dactive::VB
    gsc::GS                       # concretely parametrized — this field feeds
    h_config::SpinConfig          # the hot per-stage gradient call (no boxing)
end

function GPULLGState(gH, prob::LLGProblem, config0::SpinConfig,
                     sigma::Vector{Float64})
    backend = gH.backend
    n = n_sites(prob.H)
    alloc3() = KernelAbstractions.allocate(backend, SVector{3,Float64}, n)
    dconfig = alloc3()
    copyto!(dconfig, config0)
    depred = alloc3()
    domega1 = alloc3()
    dG = alloc3()
    dgth = alloc3()
    fill!(dgth, zero(SVector{3,Float64}))        # deterministic runs never write it
    dgzee = alloc3()
    copyto!(dgzee, prob.gzee)
    dpref = KernelAbstractions.allocate(backend, Float64, n)
    copyto!(dpref, prob.pref)
    dalpha = KernelAbstractions.allocate(backend, Float64, n)
    copyto!(dalpha, prob.alpha)
    dsigma = KernelAbstractions.allocate(backend, Float64, n)
    copyto!(dsigma, sigma)
    dactive = KernelAbstractions.allocate(backend, Int8, n)
    copyto!(dactive, Int8[prob.H.site_active[s] ? Int8(1) : Int8(0) for s = 1:n])
    gsc = SCEMonteCarlo.GPUGradientScratch(gH)
    return GPULLGState(dconfig, depred, domega1, dG, dgth, dgzee, dpref, dalpha,
                       dsigma, dactive, gsc, SpinConfig(undef, n))
end

# Stable, validation-only identifier of a KernelAbstractions backend for the
# checkpoint (`run/backend`); the actual backend object always comes from the
# caller's `gH`, never from the file.
_backend_tag(::KernelAbstractions.CPU)::String = "cpu"
_backend_tag(backend)::String = lowercase(string(nameof(typeof(backend))))
