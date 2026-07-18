# The GPU driver: `run_llg_gpu` + `_llg_loop_gpu!` + the GPU resume method
# (decision record docs/specs/gpu-llg.md). The loop downloads a host snapshot
# ONLY on measurement/checkpoint events (`_ck_due` mirrors `_ck_llg!`'s
# cadence); between events the per-step launches queue with no synchronization.

"""
    run_llg_gpu(prob::LLGProblem, config0::SpinConfig, gH;
                dt, nsteps, integrator = DepondtMertens(),
                observables = Observable[], temperature = nothing, kT = nothing,
                seed = nothing, measure_interval = 10, renorm_interval = 1000,
                workgroupsize = 128, checkpoint = nothing,
                checkpoint_interval = 0) -> LLGResult

[`run_llg`](@ref) on a KernelAbstractions device: the same physics, observables
(host-evaluated on a per-measurement snapshot — the identical `Observable`
contract and `LLGResult` semantics, so `equilibrium_stats` / `structure_factor`
/ checkpointing consume the result unchanged), and stateless Philox noise
stream (same `(seed, site, step)` draws as the CPU path — a same-seed CPU and
GPU run are the same stochastic realization, differing only through
gradient/rotation roundoff).

`gH` is `SCEMonteCarlo.GPUTiledHamiltonian(backend, prob.H)` — build it once
and reuse it across runs (the table upload is seconds at production sizes).

Determinism scope: the trajectory is bitwise reproducible for a fixed
(`seed`, backend, `workgroupsize`, package + Julia version) — the gradient
fold order depends on `workgroupsize` (pinned default 128), so unlike the CPU
path it is part of the contract, and CPU↔GPU trajectories are never bitwise
equal. The result's `compute` field and the checkpoint record all three.
"""
function run_llg_gpu(prob::LLGProblem, config0::SpinConfig, gH;
                     dt::Real, nsteps::Integer,
                     integrator::AbstractIntegrator = DepondtMertens(),
                     observables::Vector{Observable} = Observable[],
                     temperature = nothing, kT = nothing,
                     seed::Union{Nothing,Integer} = nothing,
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
    for s = 1:n
        H.site_active[s] || continue
        abs(norm(config0[s]) - 1) < 1e-8 || throw(ArgumentError(
            "config0[$s] is not a unit vector (|e| = $(norm(config0[s])))"))
    end
    dtf = Float64(dt)
    ns = Int(nsteps)
    mi = Int(measure_interval)
    thermo = !(temperature === nothing && kT === nothing)
    local kt::Float64, seed_u::UInt64
    if thermo
        kts = resolve_kt(temperature, kT)
        length(kts) == 1 || throw(ArgumentError(
            "run_llg_gpu takes a single temperature; got $(length(kts))"))
        kt = kts[1]
        for s = 1:n
            H.site_active[s] || continue
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
    spec = _RunSpec(prob, integrator, dtf, ns, mi, Int(renorm_interval), kt,
                    seed_u, observables, :gpu, _backend_tag(gH.backend), ws,
                    ClassicalThermostat())
    ck = _make_llg_checkpointer(checkpoint, checkpoint_interval, prob)
    sigma = thermo ? _sigma_noise(prob, kt, dtf) : zeros(n)
    st = GPULLGState(gH, prob, config0, sigma)
    tr = _make_trace(spec)
    tr.k = 1
    copyto!(st.h_config, config0)
    _measure!(tr.energies, tr.means, tr.series, observables, 1, 0.0, tr.times,
              prob, st.h_config)
    return _llg_loop_gpu!(spec, st, gH, tr, 0, ck)
end

# The device stepping loop from `step0` (already applied). Bit-identity of a
# resumed run rests on the same absolute-step purity as the CPU loop (noise
# counter, renorm cadence, measurement grid) plus the fixed (backend, ws).
function _llg_loop_gpu!(spec::_RunSpec, st::GPULLGState, gH, tr::_Trace,
                        step0::Int, ck)::LLGResult
    backend = gH.backend
    prob = spec.prob
    thermo = isfinite(spec.kt)
    ns = spec.nsteps
    mi = spec.measure_interval
    ws = spec.workgroupsize
    n = n_sites(prob.H)
    # invokelatest on the launches below: the upstream JET-barrier convention
    # (see _gpu_step! — abstract-Backend kernel unions)
    nkern = _noise_kernel!(backend, ws)
    rkern = _renorm_kernel!(backend, ws)
    for step = (step0 + 1):ns
        if thermo
            Base.invokelatest(nkern, st.dgth, st.dsigma, st.dactive, spec.seed,
                              step; ndrange = n)
        end
        _gpu_step!(spec.integrator, st, gH, spec.dt, ws)
        if spec.renorm_interval > 0 && step % spec.renorm_interval == 0
            Base.invokelatest(rkern, st.dconfig, st.dactive; ndrange = n)
        end
        meas = step % mi == 0 || step == ns
        if meas || _ck_due(ck, false)
            KernelAbstractions.synchronize(backend)
            copyto!(st.h_config, st.dconfig)
        end
        if meas
            tr.k += 1
            _measure!(tr.energies, tr.means, tr.series, spec.observables, tr.k,
                      step * spec.dt, tr.times, prob, st.h_config)
        end
        _ck_llg!(ck, spec, st.h_config, tr, step, false)
    end
    KernelAbstractions.synchronize(backend)
    copyto!(st.h_config, st.dconfig)
    _ck_llg!(ck, spec, st.h_config, tr, ns, true)
    return LLGResult(tr.times, tr.energies, tr.means, tr.series,
                     copy(st.h_config), ns, spec.dt, mi, spec.kt, spec.seed,
                     prob.H.n_active, _compute_string(spec),
                     _thermostat_string(spec.thermostat))
end

"""
    resume(path, prob::LLGProblem, gH;
           observables = Observable[], nsteps = nothing,
           allow_compute_switch = false, workgroupsize = nothing,
           checkpoint = path, checkpoint_interval = nothing) -> LLGResult

Continue a checkpointed run **on the device** `gH` was built for. For a
checkpoint written by [`run_llg_gpu`](@ref) on the SAME backend and workgroup
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
function SCEMonteCarlo.resume(path::AbstractString, prob::LLGProblem, gH;
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
    spec = _RunSpec(prob, data.integrator, data.dt, ns_t, data.mi, data.renorm,
                    data.kt, data.seed, observables, :gpu, tag, ws,
                    ClassicalThermostat())
    tr, config = _resume_trace(spec, data, prob, observables)
    interval = checkpoint_interval === nothing ? data.stored_interval :
               Int(checkpoint_interval)
    ck = _make_llg_checkpointer(checkpoint, interval, prob)
    thermo = isfinite(data.kt)
    sigma = thermo ? _sigma_noise(prob, data.kt, data.dt) : zeros(n_sites(H))
    st = GPULLGState(gH, prob, config, sigma)
    copyto!(st.h_config, config)
    return _llg_loop_gpu!(spec, st, gH, tr, data.step, ck)
end
