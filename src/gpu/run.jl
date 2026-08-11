# The GPU driver: `gpu_run_llg` + `_llg_loop_gpu!` + the GPU resume method
# (decision record docs/specs/gpu-llg.md). The loop downloads a host snapshot
# ONLY on measurement/checkpoint events (`_checkpoint_due` mirrors `_checkpoint_llg!`'s
# cadence); between events the per-step launches queue with no synchronization.

"""
    gpu_run_llg(prob::LLGProblem, config0::SpinConfig, gH;
                dt, nsteps, integrator = DepondtMertens(),
                observables = Observable[], temperature = nothing, kT = nothing,
                seed = nothing, thermostat = ClassicalThermostat(),
                measure_interval = 10, renorm_interval = 1000,
                workgroupsize = 128, checkpoint = nothing,
                checkpoint_interval = 0) -> LLGResult

[`run_llg`](@ref) on a KernelAbstractions device: the same physics, observables
(host-evaluated on a per-measurement snapshot — the identical `Observable`
contract and `LLGResult` semantics, so `equilibrium_stats` / `structure_factor`
/ checkpointing consume the result unchanged), the same `config0` entry door
(active columns validated and projected onto the sphere — see [`run_llg`](@ref)),
and stateless Philox noise
stream (same `(seed, site, step)` draws as the CPU path — a same-seed CPU and
GPU run are the same stochastic realization, differing only through
gradient/rotation roundoff).

`gH` is `SLCEMonteCarlo.GPUTiledHamiltonian(backend, prob.H)` — build it once
and reuse it across runs (the table upload is seconds at production sizes).

`thermostat` accepts `QuantumThermostat()` with the same contract as
[`run_llg`](@ref) (temperature required, `kT·dt/ħ ≤ $(_QT_MAX_TAU)`): the
stationary init always runs on the host and is uploaded once; the in-kernel
cascade filters the same white draws as the CPU path.

Determinism scope: the trajectory is bitwise reproducible for a fixed
(`seed`, backend, `workgroupsize`, package + Julia version) — the gradient
fold order depends on `workgroupsize` (pinned default 128), so unlike the CPU
path it is part of the contract, and CPU↔GPU trajectories are never bitwise
equal. The result's `compute` field and the checkpoint record all three.
"""
function gpu_run_llg(prob::LLGProblem, config0::SpinConfig, gH;
                     dt::Real, nsteps::Integer,
                     integrator::AbstractIntegrator = DepondtMertens(),
                     observables::Vector{Observable} = Observable[],
                     temperature = nothing, kT = nothing,
                     seed::Union{Nothing,Integer} = nothing,
                     thermostat::AbstractThermostat = ClassicalThermostat(),
                     measure_interval::Integer = 10,
                     renorm_interval::Integer = 1000,
                     workgroupsize::Integer = 128,
                     checkpoint::Union{Nothing,AbstractString} = nothing,
                     checkpoint_interval::Integer = 0)::LLGResult
    H = prob.H
    gH.host === H || model_fingerprint(gH.host) == model_fingerprint(H) ||
        throw(ArgumentError(
            "gH was built from a different Hamiltonian than prob.H"))
    n = n_sites(H)
    length(config0) == n || throw(DimensionMismatch(
        "config0 has $(length(config0)) sites but the Hamiltonian has $n"))
    (isfinite(dt) && dt > 0) || throw(ArgumentError("dt must be > 0; got $dt"))
    nsteps >= 0 || throw(ArgumentError("nsteps must be ≥ 0; got $nsteps"))
    measure_interval >= 1 ||
        throw(ArgumentError("measure_interval must be ≥ 1; got $measure_interval"))
    renorm_interval >= 0 ||
        throw(ArgumentError("renorm_interval must be ≥ 0; got $renorm_interval"))
    ws = Int(workgroupsize)
    ispow2(ws) || throw(ArgumentError("workgroupsize must be a power of two (got $ws)"))
    allunique(o.name for o in observables) ||
        throw(ArgumentError("observable names must be unique"))
    # the same entry door as run_llg — validate-then-project the active columns
    config = _config_projected(config0, H.site_has_spin)
    dtf = Float64(dt)
    ns = Int(nsteps)
    mi = Int(measure_interval)
    thermo = !(temperature === nothing && kT === nothing)
    local kt::Float64, seed_u::UInt64
    if thermo
        kts = resolve_kt(temperature, kT)
        length(kts) == 1 || throw(ArgumentError(
            "gpu_run_llg takes a single temperature; got $(length(kts))"))
        kt = kts[1]
        for s = 1:n
            H.site_has_spin[s] || continue
            prob.alpha[s] > 0 || throw(ArgumentError(
                "stochastic LLG needs α > 0 on every active site (site $s has " *
                "α = 0) — the thermal noise scales with the damping"))
        end
        ns < 2^48 || throw(ArgumentError(
            "nsteps must be < 2^48 (the noise-counter capacity); got $ns"))
        seed === nothing || seed >= 0 ||
            throw(ArgumentError("seed must be ≥ 0; got $seed"))
        seed_u = seed === nothing ? rand(UInt64) : UInt64(seed)
    else
        seed === nothing || throw(ArgumentError(
            "seed is only meaningful with a thermostat (pass temperature or kT)"))
        kt = NaN
        seed_u = 0
    end
    # host-side stationary init (the single bitwise surface), uploaded once;
    # validation shared with run_llg (one definition of the τ bound)
    fstate = _resolve_quantum_fstate(thermostat, thermo, kt, dtf, H, seed_u)
    run_spec = _RunSpec(prob, integrator, dtf, ns, mi, Int(renorm_interval), kt,
                    seed_u, observables, :gpu, _backend_tag(gH.backend), ws,
                    thermostat)
    checkpointer = _make_llg_checkpointer(checkpoint, checkpoint_interval, prob)
    sigma = thermo ? _sigma_noise(prob, kt, dtf) : zeros(n)
    st = GPULLGState(gH, prob, config, sigma, fstate)
    trace = _make_trace(run_spec)
    trace.k = 1
    copyto!(st.h_config, config)
    _measure!(trace.energies, trace.means, trace.series, observables, 1, 0.0, trace.times,
              prob, st.h_config)
    return _llg_loop_gpu!(run_spec, st, gH, trace, 0, checkpointer, fstate)
end

# The device stepping loop from `step0` (already applied). Bit-identity of a
# resumed run rests on the same absolute-step purity as the CPU loop (noise
# counter, renorm cadence, measurement grid) plus the fixed (backend, ws).
function _llg_loop_gpu!(run_spec::_RunSpec, st::GPULLGState, gH, trace::_Trace,
                        step0::Int, checkpointer,
                        fstate::Union{Nothing,_FilterState} = nothing)::LLGResult
    backend = gH.backend
    prob = run_spec.prob
    thermo = isfinite(run_spec.kt)
    ns = run_spec.nsteps
    mi = run_spec.measure_interval
    ws = run_spec.workgroupsize
    n = n_sites(prob.H)
    # invokelatest on the launches below: the upstream JET-barrier convention
    # (see _gpu_step! — abstract-Backend kernel unions)
    nkern = _noise_kernel!(backend, ws)
    qkern = _noise_kernel_quantum!(backend, ws)
    rkern = _renorm_kernel!(backend, ws)
    for step = (step0 + 1):ns
        if thermo
            if fstate === nothing
                Base.invokelatest(nkern, st.dgth, st.dsigma, st.dactive,
                                  run_spec.seed, step; ndrange = n)
            else
                Base.invokelatest(qkern, st.dgth, st.dxstate, st.dsections,
                                  st.dsigma, st.dactive, run_spec.seed, step;
                                  ndrange = n)
            end
        end
        _gpu_step!(run_spec.integrator, st, gH, run_spec.dt, ws)
        if run_spec.renorm_interval > 0 && step % run_spec.renorm_interval == 0
            Base.invokelatest(rkern, st.dconfig, st.dactive; ndrange = n)
        end
        meas = step % mi == 0 || step == ns
        ckdue = _checkpoint_due(checkpointer, false)
        if meas || ckdue
            KernelAbstractions.synchronize(backend)
            copyto!(st.h_config, st.dconfig)
            # the filter state rides the checkpoint only — measurements never
            # read it
            fstate !== nothing && ckdue && _download_filter!(fstate, st)
        end
        if meas
            trace.k += 1
            _measure!(trace.energies, trace.means, trace.series, run_spec.observables, trace.k,
                      step * run_spec.dt, trace.times, prob, st.h_config)
        end
        _checkpoint_llg!(checkpointer, run_spec, st.h_config, trace, step, false, fstate)
    end
    KernelAbstractions.synchronize(backend)
    copyto!(st.h_config, st.dconfig)
    fstate !== nothing && _download_filter!(fstate, st)
    _checkpoint_llg!(checkpointer, run_spec, st.h_config, trace, ns, true, fstate)
    return LLGResult(trace.times, trace.energies, trace.means, trace.series,
                     copy(st.h_config), ns, run_spec.dt, mi, run_spec.kt, run_spec.seed,
                     prob.b_ext, prob.H.n_active, prob.H.n_spin_active,
                     _compute_string(run_spec), _thermostat_string(run_spec.thermostat))
end

"""
    resume(path, prob::LLGProblem, gH;
           observables = Observable[], nsteps = nothing,
           allow_compute_switch = false, workgroupsize = nothing,
           checkpoint = path, checkpoint_interval = nothing) -> LLGResult

Continue a checkpointed run **on the device** `gH` was built for. For a
checkpoint written by [`gpu_run_llg`](@ref) on the SAME backend and workgroup
size, the continuation (and any `nsteps` extension) is bit-identical to the
uninterrupted device run.

Any compute switch — resuming a CPU file here, a GPU file with a different
backend tag, or a different `workgroupsize` — is refused unless
`allow_compute_switch = true`. When allowed, the continuation restores the
configuration verbatim and the same `(seed, site, step)` noise stream, so it is
the SAME physical realization continued exactly from a valid state
(statistically impeccable) — but it is **not** bit-identical to any
single-backend run; the switch must be a deliberate, loud decision (a silent
backend change once cost a walltime — see the GPU decision records).
"""
function SLCEMonteCarlo.resume(path::AbstractString, prob::LLGProblem, gH;
        observables::Vector{Observable} = Observable[],
        nsteps::Union{Nothing,Integer} = nothing,
        allow_compute_switch::Bool = false,
        workgroupsize::Union{Nothing,Integer} = nothing,
        checkpoint::Union{Nothing,AbstractString} = path,
        checkpoint_interval::Union{Nothing,Integer} = nothing)::LLGResult
    H = prob.H
    gH.host === H || model_fingerprint(gH.host) == model_fingerprint(H) ||
        throw(ArgumentError(
            "gH was built from a different Hamiltonian than prob.H"))
    data = _read_llg_ckpt(path, prob, observables)
    tag = _backend_tag(gH.backend)
    ws = workgroupsize === nothing ?
         (data.workgroupsize > 0 ? data.workgroupsize : 128) :
         Int(workgroupsize)
    ispow2(ws) || throw(ArgumentError("workgroupsize must be a power of two (got $ws)"))
    switched = data.compute != "gpu" || data.backend_tag != tag ||
               data.workgroupsize != ws
    if switched && !allow_compute_switch
        error("compute switch refused: the checkpoint was written by " *
              "(compute = \"$(data.compute)\", backend = " *
              "\"$(data.backend_tag)\", workgroupsize = " *
              "$(data.workgroupsize)) and this resume targets (\"gpu\", " *
              "\"$tag\", $ws). The continuation would be the same physical " *
              "realization but NOT bit-identical — pass " *
              "allow_compute_switch = true to proceed deliberately")
    end
    ns_t = _resume_target(data, nsteps)
    local fstate::Union{Nothing,_FilterState}
    local th::AbstractThermostat
    if data.thermostat == "quantum"
        noise_filter = _filter_from_coeffs(data.filter_coeffs)
        nlanes = 6 * length(noise_filter.sections)
        fstate = _FilterState(noise_filter, _filter_state_verbatim(data.filter_state,
                                                           nlanes,
                                                           n_sites(H)))
        th = QuantumThermostat()
    else
        fstate = nothing
        th = ClassicalThermostat()
    end
    run_spec = _RunSpec(prob, data.integrator, data.dt, ns_t, data.mi, data.renorm,
                    data.kt, data.seed, observables, :gpu, tag, ws, th)
    trace, config = _resume_trace(run_spec, data, prob, observables)
    interval = checkpoint_interval === nothing ? data.stored_interval :
               Int(checkpoint_interval)
    checkpointer = _make_llg_checkpointer(checkpoint, interval, prob)
    thermo = isfinite(data.kt)
    sigma = thermo ? _sigma_noise(prob, data.kt, data.dt) : zeros(n_sites(H))
    st = GPULLGState(gH, prob, config, sigma, fstate)
    copyto!(st.h_config, config)
    return _llg_loop_gpu!(run_spec, st, gH, trace, data.step, checkpointer, fstate)
end
