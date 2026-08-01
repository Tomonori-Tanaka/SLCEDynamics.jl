# Equilibrium statistics of a thermostatted trajectory: the bridge from
# `LLGResult.series` time series to `SLCEMonteCarlo`'s binning/jackknife machinery,
# so the SAME `Evaluable` definitions (specific heat, susceptibility, Binder …)
# work on sLLG data. Built strictly on SLCEMonteCarlo's public tier
# (`LogBinner`/`BinStore`/`bin_means`/`jackknife`/`std_error`/`tau_int`).

"""
    equilibrium_stats(result::LLGResult;
                      evaluables = standard_evaluables(),
                      discard = length(result.times) ÷ 2,
                      nbins = 32, allow_evaluables = false)
        -> Dict{Symbol,ObservableStat}

Long-time (equilibrium) averages of a **thermostatted** [`run_llg`](@ref) result:
every recorded observable series becomes an `ObservableStat` (autocorrelation-aware
mean/error/τ_int via `SLCEMonteCarlo.LogBinner`), and each `Evaluable` — the same
definitions the MC drivers accept, e.g. `SLCEMonteCarlo.standard_evaluables()` —
is jackknifed over equal-weight bins of its (scalar) input observables, with
`f(means, kT, n)` receiving the run's `kT` and the site count its **`scope`**
declares: `n_active` for `scope = :energy`, `n_spin_active` otherwise. The two
coincide on a pure-spin model and diverge exactly where a joint model has
displacement-only sites, so a `:spin`-scoped evaluable written against "the active
site count" is wrong by `n_active / n_spin_active` the first time it meets one.

- `discard`: number of leading measurements dropped as thermalization (default:
  the first half). The sLLG equilibration time scales like `1/α` — inspect the
  energy series when in doubt.
- `nbins`: jackknife bin count (a trailing remainder is dropped, as in the MC
  drivers). An evaluable needs every input among the recorded observables
  (else: `ArgumentError` — pass them to `run_llg`'s `observables`, e.g.
  `SLCEMonteCarlo.standard_observables(H)`) and ≥ 2 completed bins (else its
  stat is `NaN`).

Statistical caveat (SPEC.md): the Heun-family sLLG has weak order 1, so
equilibrium averages carry an O(dt) bias — check dt-sensitivity before trusting
a tight comparison. A deterministic run has no ensemble; calling this on one is
an error.

Quantum-thermostat runs: the equilibrium of a `QuantumThermostat` run is NOT a
Boltzmann ensemble, so fluctuation-formula evaluables (specific heat
`var(E)/kT²`, susceptibility, Binder — including the `standard_evaluables()`
default) estimate something else there and are **refused** unless
`allow_evaluables = true`. Pass `evaluables = Evaluable[]` for the raw
time-average stats (always valid), or take response functions from finite
differences across runs (e.g. specific heat from `d⟨E⟩/dT`).
"""
function equilibrium_stats(result::LLGResult;
                           evaluables::Vector{Evaluable} = standard_evaluables(),
                           discard::Integer = length(result.times) ÷ 2,
                           nbins::Integer = 32,
                           allow_evaluables::Bool = false)::Dict{Symbol,ObservableStat}
    isfinite(result.kT) || throw(ArgumentError(
        "equilibrium_stats needs a thermostatted run (run_llg with temperature " *
        "or kT); this result is a deterministic trajectory"))
    result.thermostat == "quantum" && !isempty(evaluables) && !allow_evaluables &&
        throw(ArgumentError(
            "fluctuation-formula evaluables assume a classical Boltzmann " *
            "ensemble, which a QuantumThermostat run is not — pass " *
            "evaluables = Evaluable[] for the (always valid) raw time-average " *
            "stats, take response functions from finite differences across " *
            "runs (specific heat from d⟨E⟩/dT), or insist with " *
            "allow_evaluables = true"))
    nm = length(result.times)
    0 <= discard < nm || throw(ArgumentError(
        "discard must be in [0, $(nm - 1)]; got $discard"))
    nbins >= 2 || throw(ArgumentError("nbins must be ≥ 2; got $nbins"))
    kept = nm - discard
    bin_size = max(1, fld(kept, nbins))

    stats = Dict{Symbol,ObservableStat}()
    stores = Dict{Symbol,BinStore}()
    ncomps = Dict{Symbol,Int}()
    for (name, mat) in result.series
        ncomp = size(mat, 1)
        binner = LogBinner(ncomp)
        store = BinStore(ncomp, bin_size, nbins)
        for k = (discard + 1):nm
            col = @view mat[:, k]
            push!(binner, col)
            push!(store, col)
        end
        stats[name] = ObservableStat(name, mean(binner), std_error(binner),
                                     tau_int(binner), binner.n)
        stores[name] = store
        ncomps[name] = ncomp
    end

    for ev in evaluables
        cols = Vector{Vector{Float64}}(undef, length(ev.inputs))
        nb = typemax(Int)
        ok = true
        for (q, name) in enumerate(ev.inputs)
            haskey(stores, name) || throw(ArgumentError(
                "evaluable $(ev.name) needs observable :$name, which was not " *
                "recorded — pass it to run_llg's `observables`"))
            ncomps[name] == 1 || throw(ArgumentError(
                "evaluable $(ev.name) input :$name is not a scalar observable"))
            cols[q] = vec(bin_means(stores[name]))
            nb = min(nb, length(cols[q]))
            ok &= nb >= 2
        end
        if !ok
            stats[ev.name] = ObservableStat(ev.name, [NaN], [NaN], [NaN], 0)
            continue
        end
        keys_tuple = Tuple(ev.inputs)
        # Honour the evaluable's `scope`, as the MC side does: a magnetization
        # quantity is only intensive when divided by the sites that CARRY a moment.
        # This used to pass `n_active` unconditionally — invisible on a pure-spin
        # model, where the two counts coincide, and wrong by their ratio on a joint
        # Hamiltonian with displacement-only sites.
        n = ev.scope === :energy ? result.n_active : result.n_spin_active
        f = (ms...) -> ev.f(NamedTuple{keys_tuple}(ms), result.kT, n)
        est, err = jackknife(f, cols)
        stats[ev.name] = ObservableStat(ev.name, [est], [err], [NaN], nb)
    end
    return stats
end
