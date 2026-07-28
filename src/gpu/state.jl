# Device-resident LLG state (decision record docs/specs/gpu-llg.md). SD owns
# its device arrays (`SLCEMonteCarlo`'s `GPUChainState` carries MC bookkeeping
# dynamics has no use for); the upstream contract is the shared
# `GPUTiledHamiltonian`, the stateless `gpu_energy_gradient!` entry point with
# its `GPUGradientScratch`, and plain `copyto!` transfers.

"""
    GPULLGState(gH::GPUTiledHamiltonian, prob::LLGProblem, config0::SpinConfig,
                sigma::Vector{Float64}, fstate = nothing) -> GPULLGState

Device arrays for one [`gpu_run_llg`](@ref): the configuration and the
integrator scratch (`depred`/`domega1`/`dG`/`dgth` — the device sibling of
`_LLGScratch`), the per-site problem parameters uploaded once
(`dgzee`/`dpref`/`dalpha`/`dsigma`/`dactive`), the upstream gradient scratch,
and a preallocated host snapshot buffer for measurement/checkpoint downloads.

With a quantum-thermostat `fstate::_FilterState`, the filter state lives on the
device as `dxstate` — the `n × (6·NS)` TRANSPOSE of the host `(6·NS) × n`
layout (site index fastest per lane — coalesced for the one-thread-per-site
noise kernel); the transpose happens only at the event-gated uploads/downloads
(a bitwise-lossless permutation, staged through `h_xstage`). Classical runs
carry empty (`0`-sized) filter arrays.
"""
struct GPULLGState{VC<:AbstractVector{SVector{3,Float64}},
                   VF<:AbstractVector{Float64},VB<:AbstractVector{Int8},
                   GS<:SLCEMonteCarlo.GPUGradientScratch,
                   MX<:AbstractMatrix{Float64},VS<:AbstractVector{_Biquad}}
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
    dxstate::MX                   # n × 6NS device filter state (0 × 0 classical)
    dsections::VS                 # NS device biquads (isbits; empty classical)
    h_xstage::Matrix{Float64}     # n × 6NS transfer staging (0 × 0 classical)
end

function GPULLGState(gH, prob::LLGProblem, config0::SpinConfig,
                     sigma::Vector{Float64},
                     fstate::Union{Nothing,_FilterState} = nothing)
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
    gsc = SLCEMonteCarlo.GPUGradientScratch(gH)
    if fstate === nothing
        dxstate = KernelAbstractions.allocate(backend, Float64, 0, 0)
        dsections = KernelAbstractions.allocate(backend, _Biquad, 0)
        h_xstage = Matrix{Float64}(undef, 0, 0)
    else
        nlanes = size(fstate.x, 1)
        h_xstage = Matrix{Float64}(undef, n, nlanes)
        for lane = 1:nlanes, s = 1:n
            h_xstage[s, lane] = fstate.x[lane, s]
        end
        dxstate = KernelAbstractions.allocate(backend, Float64, n, nlanes)
        copyto!(dxstate, h_xstage)
        dsections = KernelAbstractions.allocate(backend, _Biquad,
                                                length(fstate.filter.sections))
        copyto!(dsections, fstate.filter.sections)
    end
    return GPULLGState(dconfig, depred, domega1, dG, dgth, dgzee, dpref, dalpha,
                       dsigma, dactive, gsc, SpinConfig(undef, n), dxstate,
                       dsections, h_xstage)
end

# Event-gated filter-state download: device (n × 6NS) → host `_FilterState.x`
# ((6NS) × n), staged and transposed — a bitwise-lossless permutation. Caller
# has already synchronized the backend.
function _download_filter!(fstate::_FilterState, st::GPULLGState)::Nothing
    copyto!(st.h_xstage, st.dxstate)
    nlanes = size(fstate.x, 1)
    @inbounds for s = 1:size(fstate.x, 2), lane = 1:nlanes
        fstate.x[lane, s] = st.h_xstage[s, lane]
    end
    return nothing
end

# Stable, validation-only identifier of a KernelAbstractions backend for the
# checkpoint (`run/backend`); the actual backend object always comes from the
# caller's `gH`, never from the file.
_backend_tag(::KernelAbstractions.CPU)::String = "cpu"
_backend_tag(backend)::String = lowercase(string(nameof(typeof(backend))))
