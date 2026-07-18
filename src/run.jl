# The time-evolution driver and its result container.

"""
    LLGResult

Result of one [`run_llg`](@ref). Measurements are taken at step 0, every
`measure_interval` steps, and always at the final step, so `times[end] ==
nsteps·dt` and the last measurement describes exactly the returned `config`.
Fields:

- `times` [fs], `energies` — the dynamical energy (SCE + Zeeman),
- `mean_spins` — the unweighted mean direction over **active** sites (the
  `SCEMonteCarlo` magnetization convention),
- `series::Dict{Symbol,Matrix{Float64}}` — one `ncomp × n_measurements` matrix
  per user observable (see `run_llg`'s `observables`), columns aligned with
  `times`,
- `config` — the final configuration, plus the run parameters. For a
  thermostatted run, `kT` [eV] and the `seed` are recorded (`kT = NaN`,
  `seed = 0` for a deterministic run); `n_active` is the active-site count
  (what per-site [`equilibrium_stats`](@ref) evaluables normalize by);
  `compute` is the provenance tag (`"cpu"`, or `"gpu:<backend>"` from
  `run_llg_gpu` — GPU trajectories are bit-reproducible only for a fixed
  backend and workgroup size).
"""
struct LLGResult
    times::Vector{Float64}
    energies::Vector{Float64}
    mean_spins::Vector{SVector{3,Float64}}
    series::Dict{Symbol,Matrix{Float64}}
    config::SpinConfig
    nsteps::Int
    dt::Float64
    measure_interval::Int
    kT::Float64
    seed::UInt64
    n_active::Int
    compute::String
end

function Base.show(io::IO, r::LLGResult)
    therm = isfinite(r.kT) ? ", kT = $(r.kT)" : ""
    tag = r.compute == "cpu" ? "" : ", $(r.compute)"
    print(io, "LLGResult($(r.nsteps) steps, dt = $(r.dt) fs$(therm)$(tag), ",
          "$(length(r.times)) measurements, E_end = $(r.energies[end]))")
end

# The resolved, trajectory-defining parameters of one run (everything the
# continuation of a checkpointed run must reproduce verbatim). `kt = NaN` marks
# a deterministic run. On the CPU path `ntasks` stays outside (bit-identical for
# any value); on the GPU path `compute`/`backend_tag`/`workgroupsize` ARE
# trajectory-defining (the gradient fold order depends on them) and therefore
# live here and in the checkpoint (`:cpu` runs carry `("", 0)`).
struct _RunSpec
    prob::LLGProblem
    integrator::AbstractIntegrator
    dt::Float64
    nsteps::Int
    measure_interval::Int
    renorm_interval::Int
    kt::Float64
    seed::UInt64
    observables::Vector{Observable}
    compute::Symbol                  # :cpu | :gpu
    backend_tag::String              # "" on the CPU path
    workgroupsize::Int               # 0 on the CPU path
end

_compute_string(spec::_RunSpec)::String =
    spec.compute === :cpu ? "cpu" : "gpu:" * spec.backend_tag

# The measurement record under construction: the LLGResult arrays plus the count
# of columns filled so far (what a mid-run checkpoint persists).
mutable struct _Trace
    const times::Vector{Float64}
    const energies::Vector{Float64}
    const means::Vector{SVector{3,Float64}}
    const series::Dict{Symbol,Matrix{Float64}}
    k::Int
end

# measurement count of an `nsteps`-step run: step 0, every mi, plus the final
# step when it is off the grid
_nmeas(nsteps::Int, mi::Int)::Int =
    1 + div(nsteps, mi) + (nsteps % mi == 0 ? 0 : 1)

function _make_trace(spec::_RunSpec)::_Trace
    nmeas = _nmeas(spec.nsteps, spec.measure_interval)
    return _Trace(Vector{Float64}(undef, nmeas), Vector{Float64}(undef, nmeas),
                  Vector{SVector{3,Float64}}(undef, nmeas),
                  Dict{Symbol,Matrix{Float64}}(
                      o.name => Matrix{Float64}(undef, o.ncomp, nmeas)
                      for o in spec.observables), 0)
end

"""
    run_llg(prob::LLGProblem, config0::SpinConfig; dt, nsteps,
            integrator = DepondtMertens(), observables = Observable[],
            temperature = nothing, kT = nothing, seed = nothing,
            measure_interval = 10, ntasks = 1, renorm_interval = 1000,
            checkpoint = nothing, checkpoint_interval = 0)
        -> LLGResult

Integrate the LLG for `nsteps` fixed steps of `dt` [fs] from `config0` (unit
vectors; not mutated) — deterministic (`T = 0`) by default, **stochastic LLG**
when a temperature is given. Observables are recorded at step 0, every
`measure_interval` steps, and always at the final step (so the last measurement
matches the returned `config` exactly).

- `temperature` [K] XOR `kT` [model energy units] (scalar; both omitted =
  deterministic): switches on the thermal field `G_th,i = σ_i·ξ_i` with
  `σ_i = √(2 α_i kB T ħ magmom_i/(g_i Δt))` — the fluctuation–dissipation
  amplitude for this parametrization (no `(1+α²)`; see `noise.jl`). Requires
  `α > 0` on every active site (noise scales with the damping; an undamped spin
  never thermalizes). One Gaussian 3-vector per active site per step, shared by
  both integrator stages (Stratonovich). To feed [`equilibrium_stats`](@ref)'s
  default evaluables (specific heat &c.), record
  `SCEMonteCarlo.standard_observables(prob.H)` here.
- `seed`: the noise seed (thermostatted runs only; default: a fresh
  `rand(UInt64)`, recorded in the result). Draws are keyed counter-based Philox
  — the trajectory is a pure function of
  (`prob`, `config0`, `dt`, `nsteps`, `integrator`, `seed`), no RNG state
  exists, and results are bit-identical for any `ntasks`.
- `integrator`: [`DepondtMertens`](@ref) (default) or [`HeunProjected`](@ref).
- `observables`: a vector of `SCEMonteCarlo.Observable`s — the SAME definitions
  the Monte Carlo drivers accept (`Observable(name, ncomp, f)` with
  `f(config, energy, H)`, including `SCEMonteCarlo.standard_observables(H)` and
  any user-defined ones). Per the `Observable` contract, `energy` is the **SCE**
  energy (model units, intercept excluded, Zeeman not included) so a definition
  measures identical values on the same configuration in both packages. Each
  observable's time series lands in `LLGResult.series[name]` as an
  `ncomp × n_measurements` matrix. Names must be unique.
- `ntasks`: task count for the two per-step field evaluations
  (`SCEMonteCarlo.energy_gradient!`) — bit-identical results for any value.
- `renorm_interval`: every that many steps, active spins are renormalized to unit
  length (`0` disables). Depondt–Mertens preserves norms to rounding, so this only
  caps the `~ε√nsteps` rounding walk of very long runs; `HeunProjected`
  renormalizes every step by construction.
- `checkpoint`: a JLD2 path for crash-safe restart files (`nothing` disables).
  With a path, a checkpoint is written every `checkpoint_interval` steps
  (`0` ⇒ only at completion) and always once at completion; continue or reload
  with [`resume`](@ref). The noise is a stateless pure function of
  `(seed, site, step)`, so the file carries no RNG state — a resumed run is
  bit-identical to the uninterrupted one. Writes are atomic (temp file + `mv`).

A deterministic run consumes no RNG at all. Inactive sites stay bitwise frozen
throughout (no noise, no Zeeman).
"""
function run_llg(prob::LLGProblem, config0::SpinConfig; dt::Real, nsteps::Integer,
                 integrator::AbstractIntegrator = DepondtMertens(),
                 observables::Vector{Observable} = Observable[],
                 temperature = nothing, kT = nothing,
                 seed::Union{Nothing,Integer} = nothing,
                 measure_interval::Integer = 10, ntasks::Integer = 1,
                 renorm_interval::Integer = 1000,
                 checkpoint::Union{Nothing,AbstractString} = nothing,
                 checkpoint_interval::Integer = 0)::LLGResult
    H = prob.H
    n = n_sites(H)
    length(config0) == n || throw(DimensionMismatch(
        "config0 has $(length(config0)) sites but the Hamiltonian has $n"))
    (isfinite(dt) && dt > 0) || throw(ArgumentError("dt must be > 0; got $dt"))
    nsteps >= 0 || throw(ArgumentError("nsteps must be ≥ 0; got $nsteps"))
    ntasks >= 1 || throw(ArgumentError("ntasks must be ≥ 1; got $ntasks"))
    measure_interval >= 1 ||
        throw(ArgumentError("measure_interval must be ≥ 1; got $measure_interval"))
    renorm_interval >= 0 ||
        throw(ArgumentError("renorm_interval must be ≥ 0; got $renorm_interval"))
    allunique(o.name for o in observables) ||
        throw(ArgumentError("observable names must be unique"))
    for s = 1:n
        H.site_active[s] || continue
        abs(norm(config0[s]) - 1) < 1e-8 || throw(ArgumentError(
            "config0[$s] is not a unit vector (|e| = $(norm(config0[s])))"))
    end

    config = copy(config0)
    dtf = Float64(dt)
    ns = Int(nsteps)
    mi = Int(measure_interval)

    # thermostat resolution (sLLG when a temperature is given)
    thermo = !(temperature === nothing && kT === nothing)
    local kt::Float64, seed_u::UInt64
    if thermo
        kts = resolve_kt(temperature, kT)
        length(kts) == 1 || throw(ArgumentError(
            "run_llg takes a single temperature; got $(length(kts))"))
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
                    seed_u, observables, :cpu, "", 0)
    ck = _make_llg_checkpointer(checkpoint, checkpoint_interval, prob)
    tr = _make_trace(spec)
    tr.k = 1
    _measure!(tr.energies, tr.means, tr.series, observables, 1, 0.0, tr.times,
              prob, config)
    return _llg_loop!(spec, config, tr, 0, Int(ntasks), ck)
end

# The stepping loop from `step0` (already applied) to `spec.nsteps`, shared by
# `run_llg` and `resume`. `tr` holds the measurements up to and including step
# `step0`'s grid position; bit-identity of a resumed run rests on every per-step
# effect being a pure function of the absolute step index (the Philox noise
# counter, the renormalization cadence, the measurement grid).
# (`ck` is `nothing` or a `_LLGCheckpointer` — defined in checkpoint.jl, which
# is included after this file, so the annotation stays off.)
function _llg_loop!(spec::_RunSpec, config::SpinConfig, tr::_Trace, step0::Int,
                    ntasks::Int, ck)::LLGResult
    prob = spec.prob
    H = prob.H
    sc = _LLGScratch(n_sites(H))
    thermo = isfinite(spec.kt)
    sigma = thermo ? _sigma_noise(prob, spec.kt, spec.dt) : Float64[]
    ns = spec.nsteps
    mi = spec.measure_interval
    for step = (step0 + 1):ns
        thermo && _fill_noise!(sc.gth, H, sigma, spec.seed, step)
        _step!(spec.integrator, config, prob, spec.dt, sc, ntasks)
        if spec.renorm_interval > 0 && step % spec.renorm_interval == 0
            _renormalize_active!(H, config)
        end
        if step % mi == 0 || step == ns
            tr.k += 1
            # time = step count × dt, never accumulated
            _measure!(tr.energies, tr.means, tr.series, spec.observables, tr.k,
                      step * spec.dt, tr.times, prob, config)
        end
        _ck_llg!(ck, spec, config, tr, step, false)
    end
    _ck_llg!(ck, spec, config, tr, ns, true)      # the completion write
    return LLGResult(tr.times, tr.energies, tr.means, tr.series, config, ns,
                     spec.dt, mi, spec.kt, spec.seed, H.n_active,
                     _compute_string(spec))
end

# One measurement row: the SCE energy is computed once and shared by the
# dynamical-energy record (+ Zeeman) and every user observable (which receives it
# per the `Observable` contract — SCE energy, model units, intercept excluded).
function _measure!(energies::Vector{Float64}, means::Vector{SVector{3,Float64}},
                   series::Dict{Symbol,Matrix{Float64}},
                   observables::Vector{Observable}, k::Int, t::Float64,
                   times::Vector{Float64}, prob::LLGProblem,
                   config::SpinConfig)::Nothing
    e_sce = SCEMonteCarlo.total_energy(prob.H, config)
    times[k] = t
    energies[k] = e_sce + _zeeman_energy(prob, config)
    means[k] = _mean_active(prob.H, config)
    for o in observables
        v = o.f(config, e_sce, prob.H)
        col = @view series[o.name][:, k]
        if o.ncomp == 1
            col[1] = v
        else
            length(v) == o.ncomp || throw(DimensionMismatch(
                "observable $(o.name) returned $(length(v)) components; " *
                "declared ncomp = $(o.ncomp)"))
            col .= v
        end
    end
    return nothing
end

# Unweighted mean direction over active sites (the sibling magnetization
# convention); zero when no site is active.
function _mean_active(H::TiledHamiltonian, config::SpinConfig)::SVector{3,Float64}
    acc = zero(SVector{3,Float64})
    na = 0
    @inbounds for s = 1:n_sites(H)
        H.site_active[s] || continue
        acc += config[s]
        na += 1
    end
    return na == 0 ? acc : acc / na
end

function _renormalize_active!(H::TiledHamiltonian, config::SpinConfig)::Nothing
    @inbounds for s = 1:n_sites(H)
        H.site_active[s] || continue          # inactive spins stay bitwise frozen
        config[s] = config[s] / norm(config[s])
    end
    return nothing
end
