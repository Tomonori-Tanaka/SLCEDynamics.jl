# dev/fit_qtb_filter.jl — Q-M4: production fit of the quantum-FDT factor.
#
# θ(x) = x/(eˣ − 1), x = ħω/kT ≥ 0 (θ(0) = 1), fitted as the squared magnitude
# |H(i·x)|² of a real, stable, minimal-phase rational filter H(ŝ) in the
# dimensionless Laplace variable ŝ (docs/specs/quantum-thermostat.md, Q2). The
# runtime maps the pinned s-domain sections per run through the bilinear
# transform ŝ = c·(1 − z⁻¹)/(1 + z⁻¹), c = 2/τ, τ = kT·Δt/ħ, which gives the
# exact closed-form discrete PSD θ_fit(x_w), x_w = c·tan(ωΔt/2) — no aliasing
# by construction.
#
# Pipeline (the settled design):
#   1. AAA fit of θ(√u) in u = x² on [0, 200²], with u = 0 forced into the
#      support so S(0) = 1 exactly (barycentric interpolation is exact there).
#   2. Barycentric → poles/zeros in u (generalized-eigenvalue pencil), Froissart
#      cleanup, spectral factorization u = −ŝ² → left-half-plane factors.
#      Positive-real u-zeros (the raw fit's deep-tail sign dips) are paired into
#      imaginary-axis zero quadratics — the PSD form is then ≥ 0 everywhere.
#   3. Levenberg–Marquardt polish of ALL section coefficients on the weighted
#      misfit of |H(iξ)|² vs θ(ξ); positivity is automatic in the factored form
#      and the gain is renormalized to H(0) = 1 at every evaluation.
#   4. Canonicalization: each section replaced by its Hurwitz / minimal-phase
#      representative with the identical PSD, then the gain folded so every
#      section has exact unit DC gain (H(0) = 1 exact, also after the bilinear
#      map — the discrete DC gain of a section is (b0+b1+b2)/(1+a1+a2) = β0/α0).
#   5. Hard certificate gates + bilinear z-domain sanity sweep. The transfer
#      identity |H_d(e^{iωΔt})|² = |H(i·x_w)|² is verified in 256-bit
#      arithmetic (it is pure algebra; the Float64 z-evaluation near z = 1 at
#      small τ is ill-conditioned and reported separately as information).
#
# Deterministic: no RNG, no time/date. Run:  julia --project dev/fit_qtb_filter.jl
# The script FAILS (error) if any certificate gate is missed.

using LinearAlgebra
using Printf

# ------------------------------------------------------------------ target ---

theta(x::Float64)::Float64 = x == 0.0 ? 1.0 : x / expm1(x)

# ---------------------------------------------------- certificate constants ---

const X_REL_LO = 0.01      # relative-error band lower edge
const X_SPLIT  = 6.6       # relative/absolute band split
const X_MAX    = 200.0     # fit/certificate band upper edge
const GATE_REL = 1.5e-2    # relative-error gate on [X_REL_LO, X_SPLIT]
const GATE_ABS = 3.0e-4    # absolute-error gate on (X_SPLIT, X_MAX]
# One weight unifies both gates: W = 1/max(θ, W_FLOOR) turns "weighted error
# ≤ GATE_REL" into exactly the relative gate for θ ≥ W_FLOOR and exactly the
# absolute gate for θ < W_FLOOR.
const W_FLOOR    = GATE_ABS / GATE_REL
const NSEC_CAP   = 8                     # runtime section cap (_QT_MAX_NSECTIONS)
const TAUS       = (0.001, 0.01, 0.05, 0.1)
const AAA_ORDERS = (10, 11, 9, 12, 8)    # candidate support counts, tried in order

# ------------------------------------------------------------------- grids ---

# Fit grid in x: dense near 0 (the √u cusp of θ(√u)), log-spaced to X_MAX.
function fit_grid()::Vector{Float64}
    xs = vcat(0.0, collect(0.001:0.001:0.2),
              exp.(range(log(0.2), log(X_MAX); length = 1400)))
    xs = sort(unique(xs))
    xs[end] = X_MAX
    return xs
end

cert_rel_grid()::Vector{Float64} = collect(range(X_REL_LO, X_SPLIT; length = 4001))

function cert_abs_grid()::Vector{Float64}
    xs = exp.(range(log(X_SPLIT), log(X_MAX); length = 4001))
    xs[end] = X_MAX
    return xs[2:end]                     # the absolute band is open at X_SPLIT
end

# --------------------------------------------------------------------- AAA ---

# Barycentric weights for a fixed support set: null vector of the Loewner
# matrix (the standard AAA least-squares step). Returns (w, R) with R the
# barycentric evaluation on the full grid (exact on support points).
function _aaa_weights(U::Vector{Float64}, F::Vector{Float64}, supp::Vector{Int})
    n = length(U)
    nonsupp = [i for i = 1:n if i ∉ supp]
    A = Matrix{Float64}(undef, length(nonsupp), length(supp))
    for (c, s) in enumerate(supp), (r, i) in enumerate(nonsupp)
        A[r, c] = (F[i] - F[s]) / (U[i] - U[s])
    end
    w = svd(A).V[:, end]
    num = zeros(n)
    den = zeros(n)
    for (c, s) in enumerate(supp)
        for i = 1:n
            i == s && continue
            d = w[c] / (U[i] - U[s])
            num[i] += d * F[s]
            den[i] += d
        end
    end
    R = num ./ den
    for s in supp
        R[s] = F[s]
    end
    return w, R
end

# Greedy AAA on (U, F) with the u = 0 point forced first into the support.
function aaa(U::Vector{Float64}, F::Vector{Float64}, m::Int)
    i0 = findfirst(iszero, U)
    i0 === nothing && error("fit grid must contain u = 0")
    supp = [i0]
    w, R = _aaa_weights(U, F, supp)
    for _ = 2:m
        err = abs.(F .- R)
        err[supp] .= -1.0
        push!(supp, argmax(err))
        w, R = _aaa_weights(U, F, supp)
    end
    return supp, w, R
end

# Poles (v = w) / zeros (v = w .* f) of the barycentric rational: the standard
# AAA generalized-eigenvalue pencil (Nakatsukasa–Sète–Trefethen 2018).
function _bary_roots(us::Vector{Float64}, v::Vector{Float64})::Vector{ComplexF64}
    m = length(us)
    A = zeros(m + 1, m + 1)
    E = Matrix{Float64}(I, m + 1, m + 1)
    E[1, 1] = 0.0
    for j = 1:m
        A[1, j + 1] = v[j]
        A[j + 1, 1] = 1.0
        A[j + 1, j + 1] = us[j]
    end
    lam = eigen(A, E).values
    return ComplexF64[z for z in lam if isfinite(z) && abs(z) < 1e12]
end

# Froissart-doublet cleanup: delete pole/zero pairs that coincide to rounding
# (they cancel in the transfer function; kept, they breed axis artifacts).
function froissart!(ps::Vector{ComplexF64}, zs::Vector{ComplexF64})::Int
    removed = 0
    again = true
    while again && !isempty(ps) && !isempty(zs)
        again = false
        for (i, p) in enumerate(ps)
            j = argmin([abs(p - z) for z in zs])
            if abs(p - zs[j]) <= 1e-8 * (1.0 + abs(p))
                deleteat!(ps, i)
                deleteat!(zs, j)
                removed += 1
                again = true
                break
            end
        end
    end
    return removed
end

# --------------------------------------------- u-roots → s-domain factors ----

# Map u-domain roots (u = x² = −ŝ² on the imaginary axis) to left-half-plane
# s-domain factors: quadratics (q1, q0) ≡ ŝ² + q1·ŝ + q0 and linears a ≡ ŝ + a.
# Each u-root r contributes ŝ = ±√(−r); the LHP member is kept. Positive-real
# u-roots have both images on the imaginary axis: fatal for poles; for zeros
# they are the raw fit's tail sign dips and are PAIRED into one boundary
# quadratic ŝ² + √(r₁r₂) (degree-preserving; the LM polish re-fits the region).
function factorize_u_roots(roots::Vector{ComplexF64}, kind::Symbol)
    quads = Tuple{Float64,Float64}[]
    lins = Float64[]
    posreal = Float64[]
    for r in roots
        tol = 1e-8 * (1.0 + abs(r))
        if abs(imag(r)) <= tol                     # real u-root
            if real(r) < 0.0
                push!(lins, sqrt(-real(r)))        # real LHP s-root −√(−r)
            else
                kind === :pole && error(
                    "positive-real u-pole $(real(r)) (imaginary-axis s-pole)")
                push!(posreal, real(r))
            end
        elseif imag(r) > 0.0                       # conjugate-pair representative
            sr = sqrt(-r)
            p = real(sr) <= 0.0 ? sr : -sr
            push!(quads, (-2.0 * real(p), abs2(p)))
        end
    end
    sort!(posreal)
    dropped = 0
    if isodd(length(posreal))
        pop!(posreal)                              # deepest-tail crossing; the
        dropped = 1                                # polish absorbs the change
    end
    for i = 1:2:(length(posreal) - 1)
        push!(quads, (0.0, sqrt(posreal[i] * posreal[i + 1])))
    end
    return quads, lins, length(posreal) + dropped, dropped
end

# --------------------------------------------------------- section assembly ---

# Assemble numerator/denominator factors into second-order sections
# ([β2, β1, β0, α1, α0] per biquad) plus at most one first-order section
# ([β1, β0, α0]), pairing numerator and denominator factors by ascending
# corner frequency. Constant numerators pad the high-frequency sections.
function assemble_sections(zq, zl, pq, pl)
    zq = copy(zq); zl = sort(copy(zl)); pq = copy(pq); pl = sort(copy(pl))
    npoles = 2 * length(pq) + length(pl)
    nzeros = 2 * length(zq) + length(zl)
    nzeros <= npoles ||
        error("improper transfer function: $nzeros zeros vs $npoles poles")
    fo = nothing
    if isodd(npoles)
        isempty(pl) && error("odd pole count but no real s-pole")
        a = popfirst!(pl)
        if !isempty(zl)
            j = argmin(abs.(zl .- a))
            fo = [1.0, zl[j], a]
            deleteat!(zl, j)
        else
            fo = [0.0, 1.0, a]
        end
    end
    @assert iseven(length(pl))
    denquads = [[q1, q0] for (q1, q0) in pq]
    for i = 1:2:(length(pl) - 1)
        push!(denquads, [pl[i] + pl[i + 1], pl[i] * pl[i + 1]])
    end
    numquads = [[1.0, q1, q0] for (q1, q0) in zq]
    for i = 1:2:(length(zl) - 1)
        push!(numquads, [1.0, zl[i] + zl[i + 1], zl[i] * zl[i + 1]])
    end
    isodd(length(zl)) && push!(numquads, [0.0, 1.0, zl[end]])
    length(numquads) <= length(denquads) ||
        error("more numerator than denominator quadratics")
    corner(nq) = nq[1] > 0 ? sqrt(max(nq[3], 0.0)) :
                 (nq[2] > 0 ? nq[3] / nq[2] : Inf)
    sort!(numquads; by = corner)
    sort!(denquads; by = q -> q[2])
    while length(numquads) < length(denquads)
        push!(numquads, [0.0, 0.0, 1.0])           # constant numerator
    end
    biq = [vcat(numquads[i], denquads[i]) for i in eachindex(denquads)]
    return biq, fo
end

# ---------------------------------------------------------- PSD evaluation ---

# |H(iξ)|² in u = ξ² from the factored sections — nonnegative by construction:
# each factor is ((β0 − β2u)² + β1²u) / ((α0 − u)² + α1²u). Generic in the
# scalar type so the z-domain identity can be checked in BigFloat.
function model_psd(biq::Vector{Vector{Float64}},
                   fo::Union{Nothing,Vector{Float64}}, u::T)::T where {T<:Real}
    v = one(u)
    for s in biq
        num = (T(s[3]) - T(s[1]) * u)^2 + T(s[2])^2 * u
        den = (T(s[5]) - u)^2 + T(s[4])^2 * u
        v *= num / den
    end
    if fo !== nothing
        v *= (T(fo[1])^2 * u + T(fo[2])^2) / (u + T(fo[3])^2)
    end
    return v
end

# ------------------------------------------------------- Levenberg–Marquardt --

function pack(biq, fo)
    p = Float64[]
    for s in biq
        append!(p, s)
    end
    fo !== nothing && append!(p, fo)
    return p
end

function unpack(p::Vector{Float64}, nbq::Int, hasfo::Bool)
    biq = [p[(5i - 4):(5i)] for i = 1:nbq]
    fo = hasfo ? p[(5nbq + 1):(5nbq + 3)] : nothing
    return biq, fo
end

# Hand-rolled damped LM with central-difference Jacobian (deterministic).
function levmar!(p::Vector{Float64}, mres::Int, resid!::Function;
                 maxit::Int = 400)::Float64
    npar = length(p)
    r = Vector{Float64}(undef, mres)
    rp = similar(r); rm = similar(r); rt = similar(r)
    J = Matrix{Float64}(undef, mres, npar)
    resid!(r, p)
    cost = 0.5 * dot(r, r)
    lam = 1e-3
    for _ = 1:maxit
        for j = 1:npar
            h = 1e-6 * max(abs(p[j]), 1e-4)
            pj = p[j]
            p[j] = pj + h
            resid!(rp, p)
            p[j] = pj - h
            resid!(rm, p)
            p[j] = pj
            @inbounds for i = 1:mres
                J[i, j] = (rp[i] - rm[i]) / (2 * h)
            end
        end
        g = J' * r
        maximum(abs, g) < 1e-12 && break
        A = J' * J
        dA = max.(diag(A), 1e-12)
        stepped = false
        while true
            M = A + lam * Diagonal(dA)
            fac = cholesky(Symmetric(M); check = false)
            if !issuccess(fac)
                lam *= 10
                lam > 1e15 && return cost
                continue
            end
            dp = -(fac \ g)
            pn = p .+ dp
            resid!(rt, pn)
            cn = 0.5 * dot(rt, rt)
            if isfinite(cn) && cn < cost
                rel = (cost - cn) / max(cost, 1e-300)
                p .= pn
                r .= rt
                cost = cn
                lam = max(lam / 3, 1e-15)
                stepped = true
                rel < 1e-13 && return cost
                break
            end
            lam *= 4
            lam > 1e15 && return cost
        end
        stepped || return cost
    end
    return cost
end

function polish(biq0, fo0, U::Vector{Float64}, TH::Vector{Float64},
                W::Vector{Float64})
    nbq = length(biq0)
    hasfo = fo0 !== nothing
    p = pack(biq0, fo0)
    mres = length(U)
    function resid!(r, pv)
        b, f = unpack(pv, nbq, hasfo)
        p0 = model_psd(b, f, 0.0)
        if !(isfinite(p0) && p0 > 0.0)
            fill!(r, 1e6)
            return r
        end
        ok = true
        for i in eachindex(U)
            v = model_psd(b, f, U[i]) / p0
            if !isfinite(v)
                ok = false
                break
            end
            r[i] = W[i] * (v - TH[i])
        end
        ok || fill!(r, 1e6)
        return r
    end
    cost = levmar!(p, mres, resid!)
    b, f = unpack(p, nbq, hasfo)
    return b, f, cost
end

# ---------------------------------------------------------- canonical form ---

# Replace every section by the Hurwitz / minimal-phase representative with the
# IDENTICAL |·|² on the imaginary axis (the PSD quad Au² + Bu + C keeps A, B, C;
# the factor coefficients become √-canonical), then fold the gain so each
# section has exact unit DC gain: β0 := α0 literally, β1/β2 scaled by α0/β0.
# H(0) = 1 is then exact — every DC factor evaluates as α0²/α0² = 1.0.
function canonicalize(biq, fo)
    out = Vector{Float64}[]
    for s in biq
        β2, β1, β0, α1, α0 = s
        b2 = abs(β2)
        b0 = abs(β0)
        b1 = sqrt(max(β1^2 - 2 * β2 * β0 + 2 * b2 * b0, 0.0))
        a0 = abs(α0)
        a1 = sqrt(max(α1^2 - 2 * α0 + 2 * a0, 0.0))
        b0 > 0.0 || error("biquad numerator DC is zero — cannot pin H(0) = 1")
        g = a0 / b0
        push!(out, [b2 * g, b1 * g, a0, a1, a0])
    end
    fo2 = nothing
    if fo !== nothing
        β1, β0, α0 = fo
        b1 = abs(β1)
        b0 = abs(β0)
        a0 = abs(α0)
        b0 > 0.0 || error("first-order numerator DC is zero — cannot pin H(0) = 1")
        fo2 = [b1 * (a0 / b0), a0, a0]
    end
    return out, fo2
end

function section_roots(biq, fo)
    poles = ComplexF64[]
    zers = ComplexF64[]
    for s in biq
        β2, β1, β0, α1, α0 = s
        d = sqrt(complex(α1^2 - 4 * α0))
        push!(poles, (-α1 + d) / 2, (-α1 - d) / 2)
        if β2 != 0.0
            dz = sqrt(complex(β1^2 - 4 * β2 * β0))
            push!(zers, (-β1 + dz) / (2 * β2), (-β1 - dz) / (2 * β2))
        elseif β1 != 0.0
            push!(zers, complex(-β0 / β1))
        end
    end
    if fo !== nothing
        push!(poles, complex(-fo[3]))
        fo[1] != 0.0 && push!(zers, complex(-fo[2] / fo[1]))
    end
    return poles, zers
end

# -------------------------------------------------------------- certificate ---

function certificate(biq, fo, X::Vector{Float64})
    p0 = model_psd(biq, fo, 0.0)
    relmax = 0.0
    xrel = 0.0
    for x in cert_rel_grid()
        t = theta(x)
        e = abs(model_psd(biq, fo, x^2) - t) / t
        if e > relmax
            relmax = e
            xrel = x
        end
    end
    absmax = 0.0
    xabs = 0.0
    for x in cert_abs_grid()
        e = abs(model_psd(biq, fo, x^2) - theta(x))
        if e > absmax
            absmax = e
            xabs = x
        end
    end
    pmin = Inf
    xpmin = 0.0
    for x in vcat(X, cert_rel_grid(), cert_abs_grid())
        v = model_psd(biq, fo, x^2)
        if v < pmin
            pmin = v
            xpmin = x
        end
    end
    return (; p0, relmax, xrel, absmax, xabs, pmin, xpmin)
end

# ---------------------------------------------------------- z-domain sweep ---

# Full-biquad bilinear map ŝ = c·(1 − z⁻¹)/(1 + z⁻¹); generic scalar type.
function bilinear(biq, fo, c::T) where {T<:Real}
    secs = NTuple{5,T}[]
    for s in biq
        β2, β1, β0, α1, α0 = T(s[1]), T(s[2]), T(s[3]), T(s[4]), T(s[5])
        K = c * c
        D = K + α1 * c + α0
        push!(secs, ((β2 * K + β1 * c + β0) / D, 2 * (β0 - β2 * K) / D,
                     (β2 * K - β1 * c + β0) / D, 2 * (α0 - K) / D,
                     (K - α1 * c + α0) / D))
    end
    if fo !== nothing
        β1, β0, α0 = T(fo[1]), T(fo[2]), T(fo[3])
        push!(secs, ((β1 * c + β0) / (c + α0), (β0 - β1 * c) / (c + α0),
                     zero(T), (α0 - c) / (c + α0), zero(T)))
    end
    return secs
end

function discrete_psd(secs::Vector{NTuple{5,T}}, om::T)::T where {T<:Real}
    zi = exp(-im * om)                             # z⁻¹ on the unit circle
    v = one(T)
    for (b0, b1, b2, a1, a2) in secs
        v *= abs2(b0 + b1 * zi + b2 * zi^2) / abs2(one(T) + a1 * zi + a2 * zi^2)
    end
    return v
end

_zpole_radius(a1::Float64, a2::Float64)::Float64 =
    a1^2 - 4 * a2 < 0.0 ? sqrt(a2) :
    max(abs((-a1 + sqrt(a1^2 - 4 * a2)) / 2), abs((-a1 - sqrt(a1^2 - 4 * a2)) / 2))

# Verify per τ: (a) the transfer identity |H_d(e^{iωΔt})|² = |H(i·x_w)|²,
# x_w = c·tan(ωΔt/2), in 256-bit arithmetic (gate — pure algebra), (b) the same
# end-to-end in Float64 (report only: the z-form near z = 1 at small τ is
# intrinsically ill-conditioned in double precision), (c) all z-poles strictly
# inside the unit disk, (d) H_d(1) = 1 (BigFloat gate + Float64 report).
function zdomain_sweep(biq, fo)
    oms = vcat([1e-4, 1e-3, 1e-2],
               collect(range(0.05, Float64(pi) - 0.05; length = 25)),
               [Float64(pi) - 1e-3])
    dev_big = 0.0
    dev_f64 = 0.0
    dc_big = 0.0
    dc_f64 = Float64[]
    rmin = Inf
    rmax = 0.0
    setprecision(BigFloat, 256) do
        for tau in TAUS
            cb = 2 / BigFloat(tau)
            secb = bilinear(biq, fo, cb)
            for omf in oms
                om = BigFloat(omf)
                pd = discrete_psd(secb, om)
                xw = cb * tan(om / 2)
                pc = model_psd(biq, fo, xw * xw)
                dev_big = max(dev_big, Float64(abs(pd - pc) / pc))
            end
            hd1 = one(BigFloat)
            for (b0, b1, b2, a1, a2) in secb
                hd1 *= (b0 + b1 + b2) / (1 + a1 + a2)
            end
            dc_big = max(dc_big, Float64(abs(hd1 - 1)))

            c = 2.0 / tau
            secs = bilinear(biq, fo, c)
            for omf in oms
                pd = discrete_psd(secs, omf)
                pc = model_psd(biq, fo, (c * tan(omf / 2))^2)
                dev_f64 = max(dev_f64, abs(pd - pc) / pc)
            end
            hd1f = 1.0
            for (b0, b1, b2, a1, a2) in secs
                hd1f *= (b0 + b1 + b2) / (1 + a1 + a2)
                r = _zpole_radius(a1, a2)
                rmin = min(rmin, r)
                rmax = max(rmax, r)
            end
            push!(dc_f64, abs(hd1f - 1.0))
        end
    end
    return (; dev_big, dev_f64, dc_big, dc_f64, rmin, rmax)
end

# -------------------------------------------------------------------- gates ---

function check_gates(biq, fo, cert, zres)
    fails = String[]
    nsec = length(biq) + (fo === nothing ? 0 : 1)
    nsec <= NSEC_CAP || push!(fails, "section count $nsec > $NSEC_CAP")
    cert.relmax <= GATE_REL ||
        push!(fails, "max rel $(cert.relmax) > $GATE_REL on [$X_REL_LO, $X_SPLIT]")
    cert.absmax <= GATE_ABS ||
        push!(fails, "max abs $(cert.absmax) > $GATE_ABS on ($X_SPLIT, $X_MAX]")
    abs(cert.p0 - 1.0) <= 1e-12 || push!(fails, "|H(i0)|² − 1 = $(cert.p0 - 1.0)")
    cert.pmin > 0.0 || push!(fails, "|H|² min $(cert.pmin) ≤ 0 at x = $(cert.xpmin)")
    poles, zers = section_roots(biq, fo)
    all(p -> real(p) < 0.0, poles) || push!(fails, "pole(s) not strictly LHP")
    all(z -> real(z) <= 0.0, zers) || push!(fails, "zero(s) in the RHP")
    zres.dev_big <= 1e-12 ||
        push!(fails, "bilinear transfer identity (256-bit) dev $(zres.dev_big)")
    zres.dc_big <= 1e-12 || push!(fails, "H_d(1) − 1 (256-bit) = $(zres.dc_big)")
    zres.rmax < 1.0 || push!(fails, "z-pole radius $(zres.rmax) ≥ 1")
    return fails
end

# ------------------------------------------------------------------ report ---

function print_constants(biq, fo)
    println("const _QT_S_BIQUADS = (")
    for s in biq
        @printf("    (%.17g, %.17g, %.17g, %.17g, %.17g),\n", s...)
    end
    println(")")
    if fo === nothing
        println("const _QT_S_FIRST = nothing")
    else
        @printf("const _QT_S_FIRST = (%.17g, %.17g, %.17g)\n", fo...)
    end
end

function report(order, nfroissart, ndip, ndrop, biq, fo, cert, zres, relfit)
    nsec = length(biq) + (fo === nothing ? 0 : 1)
    npoles = 2 * length(biq) + (fo === nothing ? 0 : 1)
    println("=" ^ 76)
    println("AAA order (support points):        $order")
    println("Froissart doublets removed:        $nfroissart")
    println("tail dip u-zeros paired/dropped:   $ndip / $ndrop")
    println("total filter order (s-poles):      $npoles")
    println("sections (biquads + first-order):  $(length(biq)) + " *
            "$(fo === nothing ? 0 : 1) = $nsec (cap $NSEC_CAP)")
    @printf("max rel err  [%.2f, %.1f]:          %.3e  (gate %.1e)  at x = %.4f\n",
            X_REL_LO, X_SPLIT, cert.relmax, GATE_REL, cert.xrel)
    @printf("max abs err  (%.1f, %.0f]:           %.3e  (gate %.1e)  at x = %.3f\n",
            X_SPLIT, X_MAX, cert.absmax, GATE_ABS, cert.xabs)
    @printf("|H(i·0)|² − 1:                     %.3e  (gate 1e-12)\n",
            cert.p0 - 1.0)
    @printf("min |H|² on grid:                  %.6e  at x = %.4f  (> 0)\n",
            cert.pmin, cert.xpmin)
    @printf("|H(i·∞)|² (white tail floor):      %.6e\n",
            prod(s -> s[1]^2, biq; init = 1.0))
    @printf("AAA pre-polish max rel (cert band): %.3e\n", relfit)
    println("-" ^ 76)
    poles, zers = section_roots(biq, fo)
    println("s-domain poles:")
    for p in sort(poles; by = abs)
        @printf("    %+.12f %+.12f im\n", real(p), imag(p))
    end
    println("s-domain zeros:")
    for z in sort(zers; by = abs)
        @printf("    %+.12f %+.12f im\n", real(z), imag(z))
    end
    println("-" ^ 76)
    @printf("bilinear identity, 256-bit (gate 1e-12):   max dev %.3e\n",
            zres.dev_big)
    @printf("bilinear identity, Float64 (report):       max dev %.3e\n",
            zres.dev_f64)
    @printf("H_d(1) − 1, 256-bit (gate 1e-12):          max %.3e\n", zres.dc_big)
    print("H_d(1) − 1, Float64 per τ (report):        ")
    for (tau, d) in zip(TAUS, zres.dc_f64)
        @printf("τ=%g: %.1e  ", tau, d)
    end
    println()
    @printf("z-pole radius over τ sweep:                min %.12f  max %.12f\n",
            zres.rmin, zres.rmax)
    println("=" ^ 76)
end

# -------------------------------------------------------------------- main ---

function run_order(m::Int, X, U, TH, W)
    supp, w, R = aaa(U, TH, m)
    band = (X .>= X_REL_LO) .& (X .<= X_SPLIT)
    relfit = maximum(abs.(R[band] .- TH[band]) ./ TH[band])
    us = U[supp]
    fs = TH[supp]
    ps = _bary_roots(us, w)
    zs = _bary_roots(us, w .* fs)
    nfroissart = froissart!(ps, zs)
    pq, plin, _, _ = factorize_u_roots(ps, :pole)
    zq, zlin, ndip, ndrop = factorize_u_roots(zs, :zero)
    biq0, fo0 = assemble_sections(zq, zlin, pq, plin)
    biq, fo, _ = polish(biq0, fo0, U, TH, W)
    biq, fo = canonicalize(biq, fo)
    cert = certificate(biq, fo, X)
    zres = zdomain_sweep(biq, fo)
    fails = check_gates(biq, fo, cert, zres)
    return biq, fo, cert, zres, fails, nfroissart, ndip, ndrop, relfit
end

function main()
    X = fit_grid()
    U = X .^ 2
    TH = theta.(X)
    W = [1.0 / max(t, W_FLOOR) for t in TH]
    faillog = String[]
    for m in AAA_ORDERS
        result = try
            run_order(m, X, U, TH, W)
        catch err
            push!(faillog, "order $m: $(sprint(showerror, err))")
            println("order $m: pipeline failed — $(sprint(showerror, err))")
            continue
        end
        biq, fo, cert, zres, fails, nfroissart, ndip, ndrop, relfit = result
        if isempty(fails)
            report(m, nfroissart, ndip, ndrop, biq, fo, cert, zres, relfit)
            println("ALL CERTIFICATE GATES PASSED — pinned s-domain constants:")
            println()
            print_constants(biq, fo)
            return nothing
        end
        push!(faillog, "order $m: " * join(fails, "; "))
        println("order $m: gates missed — " * join(fails, "; "))
    end
    error("no AAA order passed the certificate:\n" * join(faillog, "\n"))
end

main()
