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
  (what per-site [`equilibrium_stats`](@ref) evaluables normalize by).
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
end

function Base.show(io::IO, r::LLGResult)
    therm = isfinite(r.kT) ? ", kT = $(r.kT)" : ""
    print(io, "LLGResult($(r.nsteps) steps, dt = $(r.dt) fs$(therm), ",
          "$(length(r.times)) measurements, E_end = $(r.energies[end]))")
end

"""
    run_llg(prob::LLGProblem, config0::SpinConfig; dt, nsteps,
            integrator = DepondtMertens(), observables = Observable[],
            temperature = nothing, kT = nothing, seed = nothing,
            measure_interval = 10, ntasks = 1, renorm_interval = 1000)
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

A deterministic run consumes no RNG at all. Inactive sites stay bitwise frozen
throughout (no noise, no Zeeman).
"""
function run_llg(prob::LLGProblem, config0::SpinConfig; dt::Real, nsteps::Integer,
                 integrator::AbstractIntegrator = DepondtMertens(),
                 observables::Vector{Observable} = Observable[],
                 temperature = nothing, kT = nothing,
                 seed::Union{Nothing,Integer} = nothing,
                 measure_interval::Integer = 10, ntasks::Integer = 1,
                 renorm_interval::Integer = 1000)::LLGResult
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
    sc = _LLGScratch(n)

    # thermostat resolution (sLLG when a temperature is given)
    thermo = !(temperature === nothing && kT === nothing)
    local kt::Float64, seed_u::UInt64, sigma::Vector{Float64}
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
        sigma = _sigma_noise(prob, kt, dtf)
    else
        seed === nothing || throw(ArgumentError(
            "seed is only meaningful with a thermostat (pass temperature or kT)"))
        kt = NaN
        seed_u = 0
        sigma = Float64[]
    end
    # measurement steps: 0, mi, 2mi, …, plus the final step when not on the grid
    nmeas = 1 + div(ns, mi) + (ns % mi == 0 ? 0 : 1)
    times = Vector{Float64}(undef, nmeas)
    energies = Vector{Float64}(undef, nmeas)
    means = Vector{SVector{3,Float64}}(undef, nmeas)
    series = Dict{Symbol,Matrix{Float64}}(
        o.name => Matrix{Float64}(undef, o.ncomp, nmeas) for o in observables)
    k = 1
    _measure!(energies, means, series, observables, 1, 0.0, times, prob, config)
    for step = 1:ns
        thermo && _fill_noise!(sc.gth, H, sigma, seed_u, step)
        _step!(integrator, config, prob, dtf, sc, Int(ntasks))
        if renorm_interval > 0 && step % renorm_interval == 0
            _renormalize_active!(H, config)
        end
        if step % mi == 0 || step == ns
            k += 1
            # time = step count × dt, never accumulated
            _measure!(energies, means, series, observables, k, step * dtf, times,
                      prob, config)
        end
    end
    return LLGResult(times, energies, means, series, config, ns, dtf, mi,
                     kt, seed_u, H.n_active)
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
