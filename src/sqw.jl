# The dynamical spin structure factor S^{αβ}(q,ω) from recorded trajectories —
# the settled conventions live in SPEC.md ("S(q,ω)"). One-line summary of the
# estimator (all signs and normalizations are load-bearing and gate-pinned):
#
#     s^α(q,t)     = (1/√N) Σ_{s active} e^{−2πi f·x_s} (e_s^α(t) − ē_s^α)
#     X^α(q,ω_k)   = Σ_n w_n s^α(q,t_n) e^{−2πikn/M}
#     S^{αβ}(q,ω_k)= (Δt_s/(M·W₂)) X^α(q,ω_k)^* X^β(q,ω_k)
#     S_el^{αβ}(q) = m̄^α(q)^* m̄^β(q),  m̄ = (1/√N) Σ_s e^{−2πi f·x_s} ē_s
#
# with f the fractional q (training-cell r.l.u.), x_s = cell + frac_atom the
# fractional site position, ē_s the per-site time mean over the analysis window
# (ONE global mean — never per segment), N the active-site count, and the
# two-sided fftshifted frequency axis ω_k = 2πk/(MΔt_s). A spin precessing
# positively about +ẑ lands at +ω; a magnon running toward +q lands at +q.

# --- trajectory recording -------------------------------------------------------------

"""
    trajectory_observable(H::TiledHamiltonian; name = :spins) -> Observable

An observable recording the full spin configuration (`ncomp = 3·n_sites(H)`,
column-major `[x₁,y₁,z₁, x₂,…]` = `vec(to_matrix(config))`). Feed it to
[`run_llg`](@ref)'s `observables` to acquire the trajectory that
[`structure_factor`](@ref) consumes; the series is persisted by the checkpoint
machinery like any other observable (note the file-size cost: 24 bytes per site
per measurement). Memory guidance: the in-RAM trajectory is `24·n_sites·n_meas`
bytes — keep it ≲ 10 GB (e.g. 8³ cells × 68 atoms × 4096 frames ≈ 3.4 GB).
"""
function trajectory_observable(H::TiledHamiltonian;
                               name::Symbol = :spins)::Observable
    return Observable(name, 3 * n_sites(H),
                      (config, _, _) -> vec(SCEMonteCarlo.to_matrix(config)))
end

"""
    trajectory(res::LLGResult; name = :spins)
        -> (; traj::AbstractArray{Float64,3}, times::Vector{Float64})

The recorded trajectory as a zero-copy `3 × n_sites × n_meas` reshape of
`res.series[name]`, with its time axis. When the run's final measurement was off
the `measure_interval` grid (`nsteps % measure_interval ≠ 0`), that last column
is dropped — the S(q,ω) estimator needs a strictly uniform time grid.
"""
function trajectory(res::LLGResult; name::Symbol = :spins)
    haskey(res.series, name) || throw(ArgumentError(
        "no series :$name — pass trajectory_observable(H) to run_llg's " *
        "observables"))
    mat = res.series[name]
    size(mat, 1) % 3 == 0 || throw(ArgumentError(
        "series :$name has $(size(mat, 1)) components — not a 3n trajectory"))
    times = res.times
    nt = length(times)
    if nt >= 3 && !isapprox(times[nt] - times[nt-1], times[2] - times[1];
                            rtol = 1e-8)
        nt -= 1                       # off-grid final measurement
    end
    traj = reshape(view(mat, :, 1:nt), 3, size(mat, 1) ÷ 3, nt)
    return (; traj, times = times[1:nt])
end

# --- q handling -----------------------------------------------------------------------

# Integer representation m_i = f_i·N_i of a commensurate fractional q (throws on
# an incommensurate one — snapping is q_path's job, never this layer's).
function _q_ints(f::SVector{3,Float64}, dims::NTuple{3,Int})::NTuple{3,Int}
    m = ntuple(i -> round(Int, f[i] * dims[i]), 3)
    for i = 1:3
        abs(f[i] * dims[i] - m[i]) <= 1e-8 || throw(ArgumentError(
            "q = $(Tuple(f)) is not commensurate with dims = $dims along axis " *
            "$i; the nearest representable component is $(m[i])/$(dims[i]) — " *
            "use q_path(...; dims) to snap"))
    end
    return m
end

# Cartesian wavevector [rad/Å]: q_cart = 2π·Bᵀ·f with B = crystal.lattice.reciprocal
# (rows bᵢ, aᵢ·bⱼ = δᵢⱼ, NO 2π upstream — the 2π enters here exactly once).
_q_cartesian(crystal::Crystal, f::SVector{3,Float64})::SVector{3,Float64} =
    SVector{3,Float64}(2π * (crystal.lattice.reciprocal' * f))

# Site phases φ_s = e^{−2πi f·x_s} for one q (integer form m). The cell part of
# f·x_s is computed in exact integer arithmetic mod 1; the per-atom fractional
# part is one Float64 product per atom (identical for every cell copy; `atom_ph`
# is caller-provided scratch of length n_cell_atoms). The spatial sign
# convention (−) is THE phase-sign site — pinned by the ring-gate +q/−q
# asymmetry and the translation-covariance gate in test_sqw_core.jl.
function _fill_phases!(phi::Vector{ComplexF64}, atom_ph::Vector{Float64},
                       H::TiledHamiltonian, crystal::Crystal,
                       m::NTuple{3,Int})::Nothing
    n_a = H.n_cell_atoms
    N1, N2, N3 = H.dims
    den = N1 * N2 * N3
    length(atom_ph) == n_a || throw(DimensionMismatch(
        "atom_ph scratch has length $(length(atom_ph)); expected $n_a"))
    for a = 1:n_a
        atom_ph[a] = (m[1] / N1) * crystal.frac_positions[1, a] +
                     (m[2] / N2) * crystal.frac_positions[2, a] +
                     (m[3] / N3) * crystal.frac_positions[3, a]
    end
    @inbounds for s = 1:n_sites(H)
        a = mod1(s, n_a)                      # ≡ site_atom(H, s) (upstream pin)
        cl = (s - 1) ÷ n_a
        c1 = cl % N1
        c2 = (cl ÷ N1) % N2
        c3 = cl ÷ (N1 * N2)
        num = m[1] * c1 * N2 * N3 + m[2] * c2 * N1 * N3 + m[3] * c3 * N1 * N2
        phi[s] = cis(-2π * (mod(num, den) / den + atom_ph[a]))
    end
    return nothing
end

"""
    q_path(crystal::Crystal, vertices; npoints = 100, dims = nothing)
        -> (; qs, qs_requested, x, vert_idx)

Fractional q points (training-cell r.l.u.) along the piecewise-linear path
through `vertices`, distributed proportionally to Cartesian segment length.
`x` is the cumulative Cartesian path length [rad/Å] of the returned points (the
plot abscissa) and `vert_idx` the indices of the path vertices among them.

With `dims` (pass `H.dims`), every point is **snapped** to the supercell's
commensurate grid (`round(fᵢNᵢ)/Nᵢ` per component; exact half-grid points
follow `round`'s ties-to-even) and consecutive duplicates are merged — `qs`
holds the snapped points, `qs_requested` the pre-snap ones, so the snapping is
always visible. Without `dims`, no snapping happens (and
[`structure_factor`](@ref) will throw on incommensurate points — by design).
"""
function q_path(crystal::Crystal,
                vertices::AbstractVector{<:AbstractVector{<:Real}};
                npoints::Integer = 100, dims = nothing)
    length(vertices) >= 2 ||
        throw(ArgumentError("q_path needs at least two vertices"))
    npoints >= length(vertices) || throw(ArgumentError(
        "npoints = $npoints is fewer than the $(length(vertices)) vertices"))
    verts = [SVector{3,Float64}(v) for v in vertices]
    seglen = [norm(_q_cartesian(crystal, verts[i+1]) -
                   _q_cartesian(crystal, verts[i]))
              for i = 1:length(verts)-1]
    total = sum(seglen)
    total > 0 || throw(ArgumentError("the path has zero Cartesian length"))
    raw = SVector{3,Float64}[]
    vpos = Int[]
    for i = 1:length(verts)-1
        ni = max(1, round(Int, (npoints - 1) * seglen[i] / total))
        push!(vpos, length(raw) + 1)
        for j = 0:ni-1
            push!(raw, verts[i] + (j / ni) * (verts[i+1] - verts[i]))
        end
    end
    push!(vpos, length(raw) + 1)
    push!(raw, verts[end])
    snapped = if dims === nothing
        copy(raw)
    else
        d = NTuple{3,Int}(dims)
        [SVector{3,Float64}(ntuple(i -> round(q[i] * d[i]) / d[i], 3))
         for q in raw]
    end
    # merge consecutive duplicates (post-snap), tracking old → new indices
    keep = [1]
    for i = 2:length(snapped)
        snapped[i] == snapped[keep[end]] || push!(keep, i)
    end
    newindex = zeros(Int, length(snapped))
    for (ni, oi) in enumerate(keep)
        newindex[oi] = ni
    end
    for i = 1:length(snapped)                # map dropped points to predecessors
        newindex[i] == 0 && (newindex[i] = newindex[i-1])
    end
    qs = snapped[keep]
    x = zeros(length(qs))
    for i = 2:length(qs)
        x[i] = x[i-1] + norm(_q_cartesian(crystal, qs[i]) -
                             _q_cartesian(crystal, qs[i-1]))
    end
    return (; qs, qs_requested = raw[keep], x, vert_idx = newindex[vpos])
end

# --- result container -----------------------------------------------------------------

"""
    SQWResult

Dynamical structure factor on a q-list × two-sided frequency grid. Fields:

- `qs` — the fractional q (training-cell r.l.u.), `qs_requested` the pre-snap
  inputs (identical unless a snapping path helper produced them), `qs_cart` the
  Cartesian wavevectors [rad/Å],
- `omegas` [rad/fs, ascending, two-sided; for even `nfft` the Nyquist bin exists
  only on the negative side] and the convenience axis `energies_mev = ħω` [meV],
- `S[α, β, iq, iω]` — the full Hermitian tensor (diagonals real ≥ 0); use the
  reductions [`sqw_diag`](@ref), [`sqw_trace`](@ref), [`sqw_perp`](@ref),
  [`sqw_plusminus`](@ref),
- `S_el[α, β, iq]` — the elastic (per-site-time-mean) tensor, separated so Bragg
  weight never leaks into the inelastic spectrum,
- `err` — per-element standard error over independent realizations (seed
  ensembles with ≥ 3 members; `nothing` otherwise). For the complex
  off-diagonals this is the combined real+imag scatter
  `√(Σ|S_r − S̄|²/(R(R−1)))` — one real number per element,
- the estimation metadata (`window`, `nfft`, `nsegments`, `overlap`, `discard`,
  `dt_meas` [fs], `nrealizations`).

Normalization: with `window = :none` and one segment,
`Σ_k S^{αα}(q,ω_k)/(nfft·dt_meas) = ⟨|s^α(q,t)|²⟩` exactly (Parseval), i.e. the
ω-integral `∫S dω/2π` recovers the equal-time fluctuation correlator.
"""
struct SQWResult
    qs::Vector{SVector{3,Float64}}
    qs_requested::Vector{SVector{3,Float64}}
    qs_cart::Vector{SVector{3,Float64}}
    omegas::Vector{Float64}
    energies_mev::Vector{Float64}
    S::Array{ComplexF64,4}
    S_el::Array{ComplexF64,3}
    err::Union{Nothing,Array{Float64,4}}
    window::Symbol
    nfft::Int
    nsegments::Int
    overlap::Float64
    discard::Int
    dt_meas::Float64
    nrealizations::Int

    function SQWResult(qs, qs_requested, qs_cart, omegas, energies_mev, S, S_el,
                       err, window, nfft, nsegments, overlap, discard, dt_meas,
                       nrealizations)
        nq = length(qs)
        nw = length(omegas)
        (length(qs_requested) == nq && length(qs_cart) == nq &&
         length(energies_mev) == nw) ||
            throw(DimensionMismatch("inconsistent q/ω axis lengths"))
        size(S) == (3, 3, nq, nw) || throw(DimensionMismatch(
            "S has size $(size(S)); expected (3, 3, $nq, $nw)"))
        size(S_el) == (3, 3, nq) || throw(DimensionMismatch(
            "S_el has size $(size(S_el)); expected (3, 3, $nq)"))
        err === nothing || size(err) == size(S) || throw(DimensionMismatch(
            "err has size $(size(err)); expected $(size(S))"))
        # structural identities of the estimator (violations = implementation bug)
        for iq = 1:nq, iw = 1:nw, β = 1:3, α = 1:β
            a = S[α, β, iq, iw]
            b = S[β, α, iq, iw]
            abs(a - conj(b)) <= 1e-10 * (1 + abs(a)) ||
                error("S is not Hermitian at (α,β,q,ω) = ($α,$β,$iq,$iw)")
            if α == β
                abs(imag(a)) <= 1e-10 * (1 + abs(a)) && real(a) >= -1e-12 ||
                    error("S diagonal is not real ≥ 0 at ($α,$iq,$iw)")
            end
        end
        return new(qs, qs_requested, qs_cart, omegas, energies_mev, S, S_el,
                   err, window, nfft, nsegments, overlap, discard, dt_meas,
                   nrealizations)
    end
end

function Base.show(io::IO, r::SQWResult)
    print(io, "SQWResult($(length(r.qs)) q × $(length(r.omegas)) ω, ",
          "window = :$(r.window), nfft = $(r.nfft), ",
          "nsegments = $(r.nsegments), nrealizations = $(r.nrealizations))")
end

"""ħω axis in eV (`HBAR_EV_FS .* omegas`)."""
homega_ev(r::SQWResult)::Vector{Float64} = HBAR_EV_FS .* r.omegas

"""ħω axis in meV (the `energies_mev` field, recomputed)."""
homega_mev(r::SQWResult)::Vector{Float64} = 1000 .* HBAR_EV_FS .* r.omegas

"""Frequency axis in THz (`1000·ω/2π`)."""
freq_thz(r::SQWResult)::Vector{Float64} = 1000 .* r.omegas ./ (2π)

# --- reductions (pure functions of the stored tensor) ---------------------------------

"""
    sqw_diag(r::SQWResult) -> Array{Float64,3}

The three diagonal components `S^{xx}, S^{yy}, S^{zz}` as `out[α, iq, iω]`.
"""
sqw_diag(r::SQWResult)::Array{Float64,3} =
    Float64[real(r.S[α, α, iq, iw]) for α = 1:3, iq = 1:length(r.qs),
            iw = 1:length(r.omegas)]

"""
    sqw_trace(r::SQWResult) -> Matrix{Float64}

`Σ_α S^{αα}` as `out[iq, iω]` — the rotation-invariant total intensity (the
sum-rule object).
"""
sqw_trace(r::SQWResult)::Matrix{Float64} =
    Float64[real(r.S[1, 1, iq, iw] + r.S[2, 2, iq, iw] + r.S[3, 3, iq, iw])
            for iq = 1:length(r.qs), iw = 1:length(r.omegas)]

"""
    sqw_perp(r::SQWResult) -> Matrix{Float64}

The unpolarized-neutron weight `Σ_{αβ}(δ_{αβ} − q̂_α q̂_β) S^{αβ}` as
`out[iq, iω]`, with `q̂` the physical Cartesian direction (including any
reciprocal-lattice offset). At `q = 0` the direction is undefined — that row is
`NaN` (no silent convention).
"""
function sqw_perp(r::SQWResult)::Matrix{Float64}
    nq, nw = length(r.qs), length(r.omegas)
    out = Matrix{Float64}(undef, nq, nw)
    for iq = 1:nq
        # exact-zero test is safe by construction: q = 0 ⇒ f = 0 ⇒ qs_cart is
        # exactly zero, and the smallest nonzero commensurate |q| is ~|b|/Nᵢ
        qn = norm(r.qs_cart[iq])
        if qn == 0
            out[iq, :] .= NaN
            continue
        end
        qh = r.qs_cart[iq] / qn
        for iw = 1:nw
            acc = 0.0
            for β = 1:3, α = 1:3
                p = (α == β ? 1.0 : 0.0) - qh[α] * qh[β]
                acc += p * real(r.S[α, β, iq, iw])
            end
            out[iq, iw] = acc
        end
    end
    return out
end

"""
    sqw_plusminus(r::SQWResult; axis = SVector(0.0, 0.0, 1.0)) -> Matrix{Float64}

The transverse spectrum `S^{+−} = S_uu + S_vv − 2·Im S_uv` (`e^± = e_u ± i e_v`
in a right-handed frame `(u, v, axis)`) as `out[iq, iω]`, real ≥ 0. A spin
precessing positively about `+axis` contributes at **positive** ω — this is the
component that resolves the magnon circulation sense. The default axis ẑ uses
exactly `u = x̂, v = ŷ`.
"""
function sqw_plusminus(r::SQWResult;
                       axis::SVector{3,Float64} = SVector(0.0, 0.0, 1.0))
    n = axis / norm(axis)
    u, v = if n == SVector(0.0, 0.0, 1.0)
        SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)
    else
        t = abs(n[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
        u0 = normalize(cross(n, t))
        u0, cross(n, u0)                      # u × v = n (right-handed)
    end
    nq, nw = length(r.qs), length(r.omegas)
    out = Matrix{Float64}(undef, nq, nw)
    for iw = 1:nw, iq = 1:nq
        suu = 0.0 + 0.0im
        svv = 0.0 + 0.0im
        suv = 0.0 + 0.0im
        for β = 1:3, α = 1:3
            sab = r.S[α, β, iq, iw]
            suu += u[α] * u[β] * sab
            svv += v[α] * v[β] * sab
            suv += u[α] * v[β] * sab
        end
        out[iq, iw] = real(suu) + real(svv) - 2 * imag(suv)
    end
    return out
end

"""
    sqw_elastic(r::SQWResult) -> Array{ComplexF64,3}

The elastic tensor `S_el[α, β, iq]` (the `S_el` field).
"""
sqw_elastic(r::SQWResult)::Array{ComplexF64,3} = r.S_el

# --- the estimator core ---------------------------------------------------------------

# Resolved estimator parameters: the analysis window [c0, c0+L-1] (columns of the
# trajectory), the derived segment length M (= nfft, power of two, truncate-only)
# and hop, and the window samples.
function _sqw_params(nt::Int, dtm::Float64, discard::Int, nsegments::Int,
                     overlap::Float64, window::Symbol,
                     seglength::Union{Nothing,Int})
    nsegments >= 1 ||
        throw(ArgumentError("nsegments must be ≥ 1; got $nsegments"))
    (0.0 <= overlap < 1.0) ||
        throw(ArgumentError("overlap must be in [0, 1); got $overlap"))
    (0 <= discard <= nt - 2) || throw(ArgumentError(
        "discard must leave ≥ 2 measurements (0 ≤ discard ≤ $(nt - 2))"))
    L = nt - discard
    hop(M) = max(1, round(Int, M * (1 - overlap)))
    local M::Int
    if seglength === nothing
        M = prevpow(2, L)
        while M >= 2 && (nsegments - 1) * hop(M) + M > L
            M >>>= 1
        end
        M >= 2 || throw(ArgumentError(
            "no power-of-two segment length fits $nsegments segments in " *
            "$L samples"))
    else
        M = Int(seglength)
        (M >= 2 && ispow2(M)) || throw(ArgumentError(
            "seglength must be a power of two ≥ 2; got $M"))
        (nsegments - 1) * hop(M) + M <= L || throw(ArgumentError(
            "$nsegments segments of length $M (overlap $overlap) need " *
            "$((nsegments - 1) * hop(M) + M) samples; only $L available"))
    end
    w = Vector{Float64}(undef, M)
    _fill_window!(w, window)
    return (; c0 = discard + 1, L, M, hop = hop(M), w, W2 = _window_power(w))
end

# One q-chunk of the estimator (thread-disjoint writes into S / S_el).
function _sqw_chunk!(S::Array{ComplexF64,4}, S_el::Array{ComplexF64,3},
                     lo::Int, hi::Int, traj, ebar::Matrix{Float64},
                     H::TiledHamiltonian, crystal::Crystal,
                     ms::Vector{NTuple{3,Int}}, active::Vector{Int},
                     p)::Nothing
    n = n_sites(H)
    invsqrtN = 1 / sqrt(length(active))
    phi = Vector{ComplexF64}(undef, n)
    atom_ph = Vector{Float64}(undef, H.n_cell_atoms)
    proj = Matrix{ComplexF64}(undef, p.L, 3)          # s^α(q, t) columns
    X = Matrix{ComplexF64}(undef, p.M, 3)
    tw = _Twiddle(p.M)
    scale = p.dtm / (p.M * p.W2 * p.nsegments)
    @inbounds for iq = lo:hi
        _fill_phases!(phi, atom_ph, H, crystal, ms[iq])
        fill!(proj, 0.0 + 0.0im)
        mbar1 = 0.0 + 0.0im
        mbar2 = 0.0 + 0.0im
        mbar3 = 0.0 + 0.0im
        for s in active                                # fixed site order
            ph = phi[s]
            e1 = ebar[1, s]
            e2 = ebar[2, s]
            e3 = ebar[3, s]
            mbar1 += ph * e1
            mbar2 += ph * e2
            mbar3 += ph * e3
            for j = 1:p.L
                col = p.c0 + j - 1
                proj[j, 1] += ph * (traj[1, s, col] - e1)
                proj[j, 2] += ph * (traj[2, s, col] - e2)
                proj[j, 3] += ph * (traj[3, s, col] - e3)
            end
        end
        mb = (mbar1 * invsqrtN, mbar2 * invsqrtN, mbar3 * invsqrtN)
        for β = 1:3, α = 1:3
            S_el[α, β, iq] = conj(mb[α]) * mb[β]
        end
        for seg = 1:p.nsegments
            off = (seg - 1) * p.hop
            for α = 1:3
                col = @view X[:, α]
                for nn = 1:p.M
                    col[nn] = p.w[nn] * invsqrtN * proj[off+nn, α]
                end
                _fft_pow2!(col, tw)
            end
            for kp = -(p.M >>> 1):(p.M>>>1)-1
                k = _dft_index(kp, p.M)
                iw = kp + (p.M >>> 1) + 1
                for β = 1:3, α = 1:3
                    S[α, β, iq, iw] += scale * conj(X[k, α]) * X[k, β]
                end
            end
        end
    end
    return nothing
end

"""
    structure_factor(traj, times, H::TiledHamiltonian, crystal::Crystal, qs;
                     nsegments = 1, overlap = 0.5, window = :hann, discard = 0,
                     seglength = nothing, ntasks = 1) -> SQWResult

The dynamical structure factor of a spin trajectory (`3 × n_sites × n_meas`,
uniform time grid `times` [fs]) at the fractional wavevectors `qs`
(training-cell r.l.u.; every `q` must be commensurate with the supercell —
`q[i]·dims[i] ∈ ℤ` — or an `ArgumentError` names the nearest representable
point; snap with [`q_path`](@ref)). `crystal` is the training cell `H` was
tiled from.

Estimator (see `SPEC.md` for the settled conventions): per-site time means over
the analysis window are always subtracted (the elastic tensor is reported
separately in `S_el`); Welch averaging over `nsegments` segments of derived
power-of-two length `nfft` (`seglength` overrides; trailing samples are
truncated, never zero-padded) with fractional `overlap` and a `:hann` (default)
or `:none` window; `discard` leading measurements are dropped as thermalization.
The result is bit-identical for any `ntasks` and across repeated calls.

Convenience methods: `structure_factor(res::LLGResult, H, crystal, qs;
name = :spins, …)` (records via [`trajectory_observable`](@ref); the off-grid
final measurement, if any, is dropped), `structure_factor(::Vector{LLGResult},
…)` (a seed ensemble: spectra averaged, realization standard errors in `err`
for ≥ 3 members), and `structure_factor(path::AbstractString, …)` (a
[`run_llg`](@ref) checkpoint file — the persisted trajectory format).
"""
function structure_factor(traj::AbstractArray{<:Real,3},
                          times::AbstractVector{<:Real}, H::TiledHamiltonian,
                          crystal::Crystal,
                          qs::AbstractVector{<:AbstractVector{<:Real}};
                          nsegments::Integer = 1, overlap::Real = 0.5,
                          window::Symbol = :hann, discard::Integer = 0,
                          seglength::Union{Nothing,Integer} = nothing,
                          ntasks::Integer = 1)::SQWResult
    n = n_sites(H)
    nt = length(times)
    size(traj) == (3, n, nt) || throw(DimensionMismatch(
        "trajectory has size $(size(traj)); expected (3, $n, $nt)"))
    nt >= 2 || throw(ArgumentError("need at least 2 measurements"))
    dtm = Float64(times[2] - times[1])
    dtm > 0 || throw(ArgumentError("times must be increasing"))
    for k = 2:nt
        isapprox(times[k] - times[k-1], dtm; rtol = 1e-8) || throw(ArgumentError(
            "times are not a uniform grid at index $k (Δt = " *
            "$(times[k] - times[k-1]) vs $dtm) — for an LLGResult, use the " *
            "LLGResult method, which drops the off-grid final measurement"))
    end
    n_atoms(crystal) == H.n_cell_atoms || throw(ArgumentError(
        "crystal has $(n_atoms(crystal)) atoms; H's training cell has " *
        "$(H.n_cell_atoms)"))
    ntasks >= 1 || throw(ArgumentError("ntasks must be ≥ 1; got $ntasks"))
    isempty(qs) && throw(ArgumentError("qs is empty"))
    qsv = [SVector{3,Float64}(q) for q in qs]
    dims = NTuple{3,Int}(H.dims)
    ms = [_q_ints(f, dims) for f in qsv]
    active = [s for s = 1:n if H.site_active[s]]
    isempty(active) && throw(ArgumentError("the Hamiltonian has no active sites"))
    p0 = _sqw_params(nt, dtm, Int(discard), Int(nsegments), Float64(overlap),
                     window, seglength === nothing ? nothing : Int(seglength))
    p = (; p0..., dtm, nsegments = Int(nsegments))
    # per-site mean over the analysis window (ONE global mean, never per segment)
    ebar = zeros(3, n)
    @inbounds for j = 1:p.L, s in active, α = 1:3
        ebar[α, s] += traj[α, s, p.c0+j-1]
    end
    ebar ./= p.L
    nq = length(qsv)
    nw = p.M
    S = zeros(ComplexF64, 3, 3, nq, nw)
    S_el = Array{ComplexF64,3}(undef, 3, 3, nq)
    ntk = min(Int(ntasks), nq)
    if ntk == 1
        _sqw_chunk!(S, S_el, 1, nq, traj, ebar, H, crystal, ms, active, p)
    else
        chunk = cld(nq, ntk)
        @sync for lo = 1:chunk:nq
            hi = min(lo + chunk - 1, nq)
            Threads.@spawn _sqw_chunk!(S, S_el, lo, hi, traj, ebar, H, crystal,
                                       ms, active, p)
        end
    end
    omegas = _freq_axis(p.M, dtm)
    return SQWResult(qsv, copy(qsv), [_q_cartesian(crystal, f) for f in qsv],
                     omegas, 1000 .* HBAR_EV_FS .* omegas, S, S_el, nothing,
                     window, p.M, Int(nsegments), Float64(overlap),
                     Int(discard), dtm, 1)
end

function structure_factor(res::LLGResult, H::TiledHamiltonian, crystal::Crystal,
                          qs::AbstractVector{<:AbstractVector{<:Real}};
                          name::Symbol = :spins, kwargs...)::SQWResult
    tr = trajectory(res; name)
    return structure_factor(tr.traj, tr.times, H, crystal, qs; kwargs...)
end

function structure_factor(results::AbstractVector{LLGResult},
                          H::TiledHamiltonian, crystal::Crystal,
                          qs::AbstractVector{<:AbstractVector{<:Real}};
                          name::Symbol = :spins, kwargs...)::SQWResult
    isempty(results) && throw(ArgumentError("results is empty"))
    R = length(results)
    first_tr = trajectory(results[1]; name)
    r1 = structure_factor(first_tr.traj, first_tr.times, H, crystal, qs;
                          kwargs...)
    R == 1 && return r1
    Ssum = copy(r1.S)
    Ssq = abs2.(r1.S)
    Sel = copy(r1.S_el)
    for r = 2:R
        tr = trajectory(results[r]; name)
        tr.times == first_tr.times || throw(ArgumentError(
            "realization $r has a different time grid — ensemble members must " *
            "share dt, nsteps, and measure_interval"))
        ri = structure_factor(tr.traj, tr.times, H, crystal, qs; kwargs...)
        Ssum .+= ri.S
        Ssq .+= abs2.(ri.S)
        Sel .+= ri.S_el
    end
    Smean = Ssum ./ R
    Sel ./= R
    err = R >= 3 ?
          sqrt.(max.(Ssq ./ R .- abs2.(Smean), 0.0) ./ (R - 1)) : nothing
    return SQWResult(r1.qs, r1.qs_requested, r1.qs_cart, r1.omegas,
                     r1.energies_mev, Smean, Sel, err, r1.window, r1.nfft,
                     r1.nsegments, r1.overlap, r1.discard, r1.dt_meas, R)
end

function structure_factor(path::AbstractString, H::TiledHamiltonian,
                          crystal::Crystal,
                          qs::AbstractVector{<:AbstractVector{<:Real}};
                          name::Symbol = :spins, kwargs...)::SQWResult
    isfile(path) || throw(ArgumentError("no checkpoint file at $path"))
    traj, times = jldopen(String(path), "r") do f
        f["kind"] == "llg" || error(
            "checkpoint kind \"$(f["kind"])\" is not an LLG run")
        f["model_fingerprint"] == model_fingerprint(H) || error(
            "checkpoint model fingerprint does not match this Hamiltonian")
        String(name) in f["run/observable_names"]::Vector{String} ||
            throw(ArgumentError(
                "checkpoint has no :$name series — it was not recorded with " *
                "trajectory_observable"))
        mat = f["trace/series/$name"]::Matrix{Float64}
        size(mat, 1) == 3 * n_sites(H) || throw(ArgumentError(
            "series :$name has $(size(mat, 1)) components; expected " *
            "$(3 * n_sites(H))"))
        (mat, f["trace/times"]::Vector{Float64})
    end
    nt = length(times)
    if nt >= 3 && !isapprox(times[nt] - times[nt-1], times[2] - times[1];
                            rtol = 1e-8)
        nt -= 1                                # off-grid final measurement
    end
    return structure_factor(reshape(view(traj, :, 1:nt), 3, n_sites(H), nt),
                            times[1:nt], H, crystal, qs; kwargs...)
end

# --- the channel-level global sum rule (verification helper) --------------------------

"""
    channel_sumrule(traj, times, H::TiledHamiltonian, crystal::Crystal)
        -> (; lhs, rhs)

The global Parseval identity of the estimator, at the sublattice-channel level
(where it is exact — the unfolded S is not periodic in one supercell BZ):
`lhs = Σ_{q ∈ grid} Σ_{a,α} Σ_k S_aa^{αα}(q,ω_k)/(M·Δt)` over the full
commensurate grid, computed through the FFT pipeline (`window = :none`, one
segment, `M = prevpow(2, n_meas)`), and `rhs = Σ_{s active} (1 − |ē_s|²)`
computed directly. They agree to ~1e-10·N for any trajectory — a plumbing gate,
not a physics assumption (unit spins enter only via `|e| = 1`).
"""
function channel_sumrule(traj::AbstractArray{<:Real,3},
                         times::AbstractVector{<:Real}, H::TiledHamiltonian,
                         crystal::Crystal)
    n = n_sites(H)
    nt = length(times)
    size(traj) == (3, n, nt) || throw(DimensionMismatch(
        "trajectory has size $(size(traj)); expected (3, $n, $nt)"))
    # (Δt cancels in Σ_k S/(MΔt), so the time grid never enters below.)
    n_a = H.n_cell_atoms
    N1, N2, N3 = H.dims
    Nc = N1 * N2 * N3
    M = prevpow(2, nt)
    tw = _Twiddle(M)
    # active atoms (activity is uniform across cells — asserted)
    act_atom = [H.site_active[a] for a = 1:n_a]
    for s = 1:n
        H.site_active[s] == act_atom[mod1(s, n_a)] ||
            error("site activity is not uniform across cells")
    end
    ebar = zeros(3, n)
    @inbounds for j = 1:M, s = 1:n, α = 1:3
        ebar[α, s] += traj[α, s, j]
    end
    ebar ./= M
    buf = Vector{ComplexF64}(undef, M)
    lhs = 0.0
    for m3 = 0:N3-1, m2 = 0:N2-1, m1 = 0:N1-1
        for a = 1:n_a
            act_atom[a] || continue
            for α = 1:3
                fill!(buf, 0.0 + 0.0im)
                for c3 = 0:N3-1, c2 = 0:N2-1, c1 = 0:N1-1
                    num = m1 * c1 * N2 * N3 + m2 * c2 * N1 * N3 +
                          m3 * c3 * N1 * N2
                    ph = cis(-2π * (mod(num, Nc) / Nc)) / sqrt(Nc)
                    s = a + n_a * (c1 + N1 * (c2 + N2 * c3))
                    for j = 1:M
                        buf[j] += ph * (traj[α, s, j] - ebar[α, s])
                    end
                end
                _fft_pow2!(buf, tw)
                lhs += sum(abs2, buf) / M^2
            end
        end
    end
    rhs = 0.0
    @inbounds for s = 1:n
        H.site_active[s] || continue
        rhs += 1 - (ebar[1, s]^2 + ebar[2, s]^2 + ebar[3, s]^2)
    end
    return (; lhs, rhs)
end
