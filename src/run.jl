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
- `config` — the final configuration, plus the run parameters.
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
end

function Base.show(io::IO, r::LLGResult)
    print(io, "LLGResult($(r.nsteps) steps, dt = $(r.dt) fs, ",
          "$(length(r.times)) measurements, E_end = $(r.energies[end]))")
end

"""
    run_llg(prob::LLGProblem, config0::SpinConfig; dt, nsteps,
            integrator = DepondtMertens(), observables = Observable[],
            measure_interval = 10, ntasks = 1, renorm_interval = 1000)
        -> LLGResult

Integrate the deterministic (`T = 0`) LLG for `nsteps` fixed steps of `dt` [fs]
from `config0` (unit vectors; not mutated). Observables are recorded at step 0,
every `measure_interval` steps, and always at the final step (so the last
measurement matches the returned `config` exactly).

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

Consumes no RNG: the trajectory is a pure function of (`prob`, `config0`, `dt`,
`nsteps`, `integrator`). Inactive sites stay bitwise frozen throughout.
"""
function run_llg(prob::LLGProblem, config0::SpinConfig; dt::Real, nsteps::Integer,
                 integrator::AbstractIntegrator = DepondtMertens(),
                 observables::Vector{Observable} = Observable[],
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
    return LLGResult(times, energies, means, series, config, ns, dtf, mi)
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
