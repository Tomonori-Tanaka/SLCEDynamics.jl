# Checkpoint / resume for `run_llg`, following the sibling's format discipline
# (SLCEMonteCarlo `checkpoint.jl` / `docs/specs/checkpoint-schema.md`): the file
# holds ONLY plain data (Int/Float64/UInt64/String and arrays thereof, in named
# JLD2 groups) — no Julia struct reconstruction — and writes are atomic (temp
# file + `mv`). No RNG state exists to store (the noise is a stateless pure
# function of `(seed, site, step)`); the ONE piece of carried per-step state is
# the quantum thermostat's filter state (`state/filter`, schema v3), restored
# verbatim like the configuration. `seed` + the completed `step` + the state
# arrays reproduce the continuation bit-exactly, and *extending* a run
# (continuing to a larger `nsteps`) is bit-identical to an uninterrupted
# longer run.

# v2 adds the compute provenance (`run/compute`, `run/backend`,
# `run/workgroupsize`) — trajectory-defining on the GPU path; v1 files are
# back-read as `compute = "cpu"`. v3 adds the thermostat (`run/thermostat`,
# always; quantum-only: `run/filter_id` — provenance, no refusal —
# `run/filter/coeffs`, the AUTHORITATIVE 5 × NS biquad snapshot resume
# rebuilds from verbatim, and `state/filter`); v1/v2 back-read as
# `thermostat = "classical"`.
const _CKPT_SCHEMA_LLG = 3

# The run-side writer state: target path, write cadence, and the cached model
# fingerprint (SLCEMonteCarlo's stable FNV-1a, the shared identity check).
mutable struct _LLGCheckpointer
    const path::String
    const interval::Int         # steps between periodic writes; 0 ⇒ completion-only
    since::Int
    const fingerprint::UInt64
end

function _make_llg_checkpointer(path::Union{Nothing,AbstractString},
                                interval::Integer, prob::LLGProblem)
    path === nothing && return nothing
    interval >= 0 ||
        throw(ArgumentError("checkpoint_interval must be ≥ 0; got $interval"))
    return _LLGCheckpointer(String(path), Int(interval), 0,
                            model_fingerprint(prob.H))
end

_integrator_name(::DepondtMertens)::String = "DepondtMertens"
_integrator_name(::HeunProjected)::String = "HeunProjected"

function _integrator_from_name(name::String)::AbstractIntegrator
    name == "DepondtMertens" && return DepondtMertens()
    name == "HeunProjected" && return HeunProjected()
    error("unknown integrator \"$name\" in the checkpoint (written by a newer " *
          "package version?)")
end

# Bitwise configuration restore. Deliberately NOT `SLCEMonteCarlo.from_matrix`,
# which renormalizes each column: an ULP-level perturbation of the state forks a
# chaotic trajectory, and resume must be bit-identical to the uninterrupted run.
function _config_verbatim(m::Matrix{Float64}, n::Int)::SpinConfig
    size(m) == (3, n) || error("checkpoint config is $(size(m, 1)) × " *
                               "$(size(m, 2)); expected 3 × $n")
    return SpinConfig([SVector{3,Float64}(m[1, s], m[2, s], m[3, s])
                       for s = 1:n])
end

function _write_ckpt_llg(ck::_LLGCheckpointer, spec::_RunSpec, config::SpinConfig,
                         tr::_Trace, step::Int,
                         fstate::Union{Nothing,_FilterState})::Nothing
    prob = spec.prob
    k = tr.k
    tmp = ck.path * ".tmp." * string(getpid())   # one writer per path assumed
    jldopen(tmp, "w") do f
        f["schema_version"] = _CKPT_SCHEMA_LLG
        f["kind"] = "llg"
        f["julia_version"] = string(VERSION)
        f["package_version"] = string(pkgversion(SLCEDynamics))
        f["model_fingerprint"] = ck.fingerprint
        f["checkpoint_interval"] = ck.interval
        # trajectory-defining problem parameters (validated == on resume; the
        # Hamiltonian itself is pinned by the fingerprint)
        f["problem/magmom"] = prob.magmom
        f["problem/alpha"] = prob.alpha
        f["problem/g"] = prob.g
        f["problem/b_ext"] = Vector{Float64}(prob.b_ext)
        f["run/dt"] = spec.dt
        f["run/nsteps"] = spec.nsteps
        f["run/measure_interval"] = spec.measure_interval
        f["run/renorm_interval"] = spec.renorm_interval
        f["run/integrator"] = _integrator_name(spec.integrator)
        f["run/kT"] = spec.kt                    # NaN ⇒ deterministic run
        f["run/seed"] = spec.seed
        f["run/compute"] = String(spec.compute)
        f["run/backend"] = spec.backend_tag
        f["run/workgroupsize"] = spec.workgroupsize
        f["run/thermostat"] = _thermostat_string(spec.thermostat)
        if fstate !== nothing
            f["run/filter_id"] = _QT_FILTER_ID
            f["run/filter/coeffs"] = _filter_coeffs(fstate.filter)
            f["state/filter"] = copy(fstate.x)
        end
        f["run/observable_names"] = String[String(o.name) for o in spec.observables]
        f["run/observable_ncomps"] = Int[o.ncomp for o in spec.observables]
        f["progress/step"] = step
        f["progress/nmeas"] = k
        f["state/config"] = SLCEMonteCarlo.to_matrix(config)
        f["trace/times"] = tr.times[1:k]
        f["trace/energies"] = tr.energies[1:k]
        f["trace/mean_spins"] = Float64[tr.means[j][row] for row = 1:3, j = 1:k]
        for o in spec.observables
            f["trace/series/$(o.name)"] = tr.series[o.name][:, 1:k]
        end
    end
    mv(tmp, ck.path; force = true)
    return nothing
end

# Per-step tick (cadence) and the unconditional completion write. Writing
# consumes no RNG (this package holds none) and never perturbs the trajectory.
_ck_llg!(::Nothing, ::_RunSpec, ::SpinConfig, ::_Trace, ::Int, ::Bool,
         fstate = nothing) = nothing
function _ck_llg!(ck::_LLGCheckpointer, spec::_RunSpec, config::SpinConfig,
                  tr::_Trace, step::Int, final::Bool,
                  fstate::Union{Nothing,_FilterState} = nothing)::Nothing
    if !final
        ck.interval > 0 || return nothing
        ck.since += 1
        ck.since >= ck.interval || return nothing
        ck.since = 0
    end
    _write_ckpt_llg(ck, spec, config, tr, step, fstate)
    return nothing
end

# Whether the NEXT `_ck_llg!` call at this step would actually write — the GPU
# loop uses this to prepare a host snapshot only when one is needed. Must mirror
# `_ck_llg!`'s cadence exactly (coupled site).
_ck_due(::Nothing, ::Bool)::Bool = false
_ck_due(ck::_LLGCheckpointer, final::Bool)::Bool =
    final || (ck.interval > 0 && ck.since + 1 >= ck.interval)

# Read and validate an LLG checkpoint eagerly, closing the file before any long
# computation starts (the resumed run typically overwrites this very path).
# Shared by the CPU and GPU resume methods.
function _read_llg_ckpt(path::AbstractString, prob::LLGProblem,
                        observables::Vector{Observable})
    isfile(path) || throw(ArgumentError("no checkpoint file at $path"))
    H = prob.H
    n = n_sites(H)
    allunique(o.name for o in observables) ||
        throw(ArgumentError("observable names must be unique"))
    return jldopen(String(path), "r") do f
        # kind before schema: an MC/PT file (schema v2 upstream) should say
        # "wrong kind", not masquerade as a package-version mismatch
        f["kind"] == "llg" || error(
            "checkpoint kind \"$(f["kind"])\" is not an LLG run — MC/PT " *
            "checkpoints resume via resume(path, H::TiledHamiltonian)")
        ver = f["schema_version"]
        ver in (1, 2, 3) || error(
            "checkpoint schema v$(ver) is not readable by this package " *
            "version (knows v1–v$(_CKPT_SCHEMA_LLG))")
        f["model_fingerprint"] == model_fingerprint(H) || error(
            "checkpoint model fingerprint does not match this LLGProblem's " *
            "Hamiltonian (different model, dims, or coefficients)")
        mm = f["problem/magmom"]::Vector{Float64}
        al = f["problem/alpha"]::Vector{Float64}
        gf = f["problem/g"]::Vector{Float64}
        bx = f["problem/b_ext"]::Vector{Float64}
        length(mm) == n || error(
            "checkpoint has $(length(mm)) sites; the Hamiltonian has $n")
        # inactive entries are unvalidated placeholders — compare active only
        (SVector{3,Float64}(bx) == prob.b_ext &&
         all(!H.site_active[s] || (mm[s] == prob.magmom[s] &&
                                   al[s] == prob.alpha[s] &&
                                   gf[s] == prob.g[s]) for s = 1:n)) || error(
            "the supplied LLGProblem's parameters (magmom/alpha/g/b_ext) do " *
            "not match the checkpoint on active sites")
        names = f["run/observable_names"]::Vector{String}
        ncomps = f["run/observable_ncomps"]::Vector{Int}
        (names == String[String(o.name) for o in observables] &&
         ncomps == Int[o.ncomp for o in observables]) || error(
            "the resumed observables (names/ncomps) do not match the " *
            "checkpoint; stored: $names with $ncomps")
        (; dt = f["run/dt"]::Float64, ns_stored = f["run/nsteps"]::Int,
         mi = f["run/measure_interval"]::Int,
         renorm = f["run/renorm_interval"]::Int,
         integrator = _integrator_from_name(f["run/integrator"]::String),
         kt = f["run/kT"]::Float64, seed = f["run/seed"]::UInt64,
         compute = ver >= 2 ? f["run/compute"]::String : "cpu",
         backend_tag = ver >= 2 ? f["run/backend"]::String : "",
         workgroupsize = ver >= 2 ? f["run/workgroupsize"]::Int : 0,
         thermostat = ver >= 3 ? f["run/thermostat"]::String : "classical",
         # quantum-only groups; sentinel empties keep the tuple type uniform
         filter_id = ver >= 3 && f["run/thermostat"] == "quantum" ?
                     f["run/filter_id"]::String : "",
         filter_coeffs = ver >= 3 && f["run/thermostat"] == "quantum" ?
                         f["run/filter/coeffs"]::Matrix{Float64} : zeros(5, 0),
         filter_state = ver >= 3 && f["run/thermostat"] == "quantum" ?
                        f["state/filter"]::Matrix{Float64} : zeros(0, 0),
         step = f["progress/step"]::Int, k = f["progress/nmeas"]::Int,
         config = _config_verbatim(f["state/config"]::Matrix{Float64}, n),
         times = f["trace/times"]::Vector{Float64},
         energies = f["trace/energies"]::Vector{Float64},
         means = f["trace/mean_spins"]::Matrix{Float64},
         series = Dict{Symbol,Matrix{Float64}}(
             o.name => f["trace/series/$(o.name)"]::Matrix{Float64}
             for o in observables),
         stored_interval = f["checkpoint_interval"]::Int)
    end
end

# Target-step resolution and the trace-prefix grid checks (shared by both
# resume methods).
function _resume_target(data, nsteps::Union{Nothing,Integer})::Int
    # a package-written file always satisfies these; fail cleanly on corruption
    (data.mi >= 1 && data.renorm >= 0 && data.step >= 0 && data.k >= 1) || error(
        "checkpoint run parameters are corrupted (measure_interval = " *
        "$(data.mi), renorm_interval = $(data.renorm), step = $(data.step), " *
        "nmeas = $(data.k))")
    ns_t = nsteps === nothing ? data.ns_stored : Int(nsteps)
    ns_t >= data.step || throw(ArgumentError(
        "nsteps = $ns_t is smaller than the checkpoint's completed step " *
        "$(data.step)"))
    if isfinite(data.kt)
        ns_t < 2^48 || throw(ArgumentError(
            "nsteps must be < 2^48 (the noise-counter capacity); got $ns_t"))
    end
    # the trace must be a prefix of the target run's measurement grid: `k` grid
    # measurements ≤ step, plus at most one off-grid final of a COMPLETED run
    grid_k = 1 + div(data.step, data.mi)
    if data.k == grid_k + 1
        (data.step == data.ns_stored && data.step % data.mi != 0) || error(
            "checkpoint trace is inconsistent ($(data.k) measurements at " *
            "step $(data.step) with measure_interval $(data.mi))")
        ns_t == data.step || throw(ArgumentError(
            "cannot extend past step $(data.step): the completed run's final " *
            "measurement at step $(data.step) is off the measurement grid " *
            "(measure_interval = $(data.mi)), so its trace is not a prefix " *
            "of the longer run's"))
    else
        data.k == grid_k || error(
            "checkpoint trace is inconsistent ($(data.k) measurements at " *
            "step $(data.step) with measure_interval $(data.mi))")
    end
    return ns_t
end

# Build the trace prefix (and the truncation-edge final measurement) for a
# resume — shared by both resume methods.
function _resume_trace(spec::_RunSpec, data, prob::LLGProblem,
                       observables::Vector{Observable})
    tr = _make_trace(spec)
    copyto!(tr.times, 1, data.times, 1, data.k)
    copyto!(tr.energies, 1, data.energies, 1, data.k)
    for j = 1:data.k
        tr.means[j] = SVector{3,Float64}(data.means[1, j], data.means[2, j],
                                         data.means[3, j])
    end
    for o in observables
        tr.series[o.name][:, 1:data.k] = data.series[o.name]
    end
    tr.k = data.k
    config = data.config
    # resuming a mid-run file to exactly its own step: the continuation loop is
    # empty, so take the (off-grid) final measurement it would otherwise record
    if spec.nsteps == data.step && tr.k < _nmeas(spec.nsteps, data.mi)
        tr.k += 1
        _measure!(tr.energies, tr.means, tr.series, observables, tr.k,
                  spec.nsteps * data.dt, tr.times, prob, config)
    end
    return tr, config
end

"""
    resume(path, prob::LLGProblem;
           observables = Observable[], nsteps = nothing, ntasks = 1,
           checkpoint = path, checkpoint_interval = nothing) -> LLGResult

Continue a checkpointed [`run_llg`](@ref) from the state saved at `path` and
return the **full** run's result — bit-identical to the uninterrupted run (the
thermal noise is a stateless function of `(seed, site, step)`, so no RNG state
exists to restore; a quantum run's filter state is restored verbatim).
Calling it on the checkpoint of a *completed* run
reconstructs that run's `LLGResult` without stepping (idempotent — safe in a
job-array retry loop). A checkpoint written by `run_llg_gpu` is refused here —
resume it with the `resume(path, prob, gH)` method.

The caller re-supplies the `LLGProblem` and the observable *functions* (closures
are not serialized); the checkpoint stores the model fingerprint, the problem
parameters (`magmom`/`alpha`/`g`/`b_ext`, compared on active sites), and the
observable names/component counts, and errors on any mismatch. Everything
trajectory-defining (`dt`, `measure_interval`, `renorm_interval`, integrator,
`kT`, `seed`, thermostat) comes from the file and cannot be overridden — a
quantum-thermostat run's filter is rebuilt from the STORED discrete
coefficients verbatim (never from the current package constants, so a future
constants re-fit cannot change a resumed trajectory) and its filter state is
restored bitwise alongside the configuration.

- `nsteps`: target total step count (default: the stored one). A value larger
  than the stored `nsteps` **extends** the run — bit-identical to a single
  uninterrupted run of that length. Extension past a completed run whose final
  measurement was off the `measure_interval` grid is refused (its recorded
  trace would not be a prefix of the longer run's).
- `checkpoint` / `checkpoint_interval`: by default the resumed run keeps
  checkpointing to the same `path` with the stored cadence (`checkpoint =
  nothing` disables; `checkpoint_interval` overrides).
"""
function SLCEMonteCarlo.resume(path::AbstractString, prob::LLGProblem;
        observables::Vector{Observable} = Observable[],
        nsteps::Union{Nothing,Integer} = nothing,
        ntasks::Integer = 1,
        checkpoint::Union{Nothing,AbstractString} = path,
        checkpoint_interval::Union{Nothing,Integer} = nothing)::LLGResult
    ntasks >= 1 || throw(ArgumentError("ntasks must be ≥ 1; got $ntasks"))
    data = _read_llg_ckpt(path, prob, observables)
    data.compute == "cpu" || error(
        "this checkpoint was written by run_llg_gpu (compute = " *
        "\"$(data.compute)\", workgroupsize = $(data.workgroupsize)) — resume " *
        "it with resume(path, prob, gH::GPUTiledHamiltonian; ...)")
    ns_t = _resume_target(data, nsteps)
    local fstate::Union{Nothing,_FilterState}
    local th::AbstractThermostat
    if data.thermostat == "quantum"
        filt = _filter_from_coeffs(data.filter_coeffs)
        nlanes = 6 * length(filt.sections)
        fstate = _FilterState(filt, _filter_state_verbatim(data.filter_state,
                                                           nlanes,
                                                           n_sites(prob.H)))
        th = QuantumThermostat()
    else
        fstate = nothing
        th = ClassicalThermostat()
    end
    spec = _RunSpec(prob, data.integrator, data.dt, ns_t, data.mi, data.renorm,
                    data.kt, data.seed, observables, :cpu, "", 0, th)
    tr, config = _resume_trace(spec, data, prob, observables)
    interval = checkpoint_interval === nothing ? data.stored_interval :
               Int(checkpoint_interval)
    ck = _make_llg_checkpointer(checkpoint, interval, prob)
    return _llg_loop!(spec, config, tr, data.step, Int(ntasks), ck, fstate)
end
