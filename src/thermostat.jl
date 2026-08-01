# The thermostat selection layer and the semi-quantum colored-noise machinery
# (decision record docs/specs/quantum-thermostat.md, settled run_spec S1–S18): the
# classical thermostat is the existing white-noise path, byte-identical to
# previous versions; the quantum thermostat filters the SAME per-step white
# draws (slots 0/1 — a same-seed classical and quantum run share one white
# realization) through a cascade of biquad sections whose discrete PSD follows
# the zero-point-free quantum FDT factor θ(x) = x/(eˣ − 1), x = ħ|ω|/kT
# (Barker & Bauer, PRB 100, 140401(R) (2019)).
#
# COUPLED SITES: `_qt_cascade!`/`_fill_noise_quantum!` ↔ the counter map below
# (and noise.jl's header) ↔ `_filter_state_space` (the same recurrence as a
# state-space system — the Lyapunov init and the test gates derive from it) ↔
# the checkpoint `state/filter` layout (schema v3) ↔ the GPU noise kernel.
# Change the recurrence, the lane layout, or the slot map and all move together.

# Hard validity bound of the quantum thermostat: τ = kT·Δt/ħ must not exceed
# this (the discrete Nyquist band must comfortably contain the support of θ;
# ≤ 1% spectral-warp accuracy at occupied modes wants τ ≲ 0.05 — enforced as a
# recommendation in the error text, not a bound).
const _QT_MAX_TAU = 0.1
const _QT_MAX_NSECTIONS = 8

# Provenance tag of the shipped filter constants, stored in quantum checkpoints
# (schema v3). Informational only — resume rebuilds the recurrence from the
# STORED coefficients, never from this package's constants, so a re-fit bumps
# this tag without invalidating old files.
const _QT_FILTER_ID = "bb-aaa10-lm-v1"

# The pinned dimensionless s-domain sections of the Barker–Bauer fit
# (dev/fit_qtb_filter.jl — deterministic, rerunning reproduces these exactly):
# |H(i·x)|² ≈ θ(x) = x/(eˣ − 1) with max rel error 4.1e-3 on x ∈ [0.01, 6.6]
# and max abs error 7.2e-5 on (6.6, 200]; H(0) = 1 EXACT (each section is
# DC-normalized: β0 ≡ α0, so every DC factor is exactly 1.0 — and survives the
# bilinear map, whose discrete DC gain per section is β0/α0). All poles/zeros
# strictly in the LHP (minimal phase, |H|² > 0 everywhere — the tail notch at
# x ≈ 13.4 bottoms at ~7e-14). Tuple order (β2, β1, β0, α1, α0) per section
# ŝ² + α1·ŝ + α0 (monic denominators). Trajectory-defining package constants:
# a re-fit changes seeded quantum trajectories (bump _QT_FILTER_ID; old
# checkpoints stay valid — they resume from their stored coefficients).
const _QT_S_BIQUADS = (
    (0.99170408200695437, 0.036502399709965036, 0.00021448034438006076,
     0.036616140853002903, 0.00021448034438006076),
    (0.91927393999329177, 0.42533407733969342, 0.034578550011187502,
     0.43847417416564211, 0.034578550011187502),
    (0.00072066283057147591, 1.9691420131595294, 2.1907633098131281,
     3.265406852584217, 2.1907633098131281),
    (0.16930356089122989, 2.5987551550330009e-05, 30.517515785768918,
     9.2962139193574611, 30.517515785768918),
)

# Lower τ bound, DERIVED from the sections above rather than pinned, so a re-fit
# of `_QT_S_BIQUADS` moves it instead of silently invalidating it.
#
# What actually degrades first is the DISCRETE DC GAIN, not the Jury margin. Each
# section is DC-normalized (β₀ ≡ α₀), so the bilinear map's per-section DC gain is
# `(b₀+b₁+b₂)/(1+a₁+a₂) = 4β₀/4α₀ = 1` exactly — but both sums are O(1)
# cancellations landing on `~α₀τ²`, so the relative error of that 1.0 is
# `eps/(α₀τ²)`. Measured against the closed form: predicted 1.035e-4 vs observed
# 1.029e-4 at τ = 1e-4, 2.59e-5 vs 2.60e-5 at 2e-4, 4.14e-6 vs 4.17e-6 at 5e-4.
# Inverting it gives the bound below. It depends only on `eps` and the smallest
# `α₀`; kT and dt enter only through τ.
#
# The Jury margin is a SEPARATE and much later failure: `1 + a₁ + a₂ → α₀·τ²`
# stops being resolvable at `α₀τ² ~ eps`, i.e. `τ ≈ 1.02e-6` — verified, a
# section is genuinely rejected as unstable there. The two are two decades apart
# and this constant is the DC one; the old comment named the Jury bound while the
# value implemented the DC one, which is why the rationale read as three decades
# off. (It also has nothing to do with the stationary-covariance conditioning —
# see `_stationary_factor`, which is solved in extended precision and is accurate
# across the whole accepted range.)
const _QT_DC_TOL = 1.0e-4        # tolerated relative degradation of H_d(1) = 1
const _QT_MIN_TAU = sqrt(eps(Float64) / (minimum(s[5] for s in _QT_S_BIQUADS) *
                                         _QT_DC_TOL))

# Working precision of the stationary-covariance solve. See `_stationary_factor`:
# the Float64 solve is not merely imprecise there, it returns a matrix that is not
# positive semi-definite, and the resulting thermal-noise power is wrong by up to
# +325 %. Measured requirement — 53 bits FAILS (the matrix is genuinely indefinite
# at that precision), 80 bits reaches 1e-8, 113 bits reaches 1e-16; 128 is that
# with 2.4× headroom.
const _QT_LYAP_PRECISION = 128

"""
    ClassicalThermostat()

White thermal noise (the classical fluctuation–dissipation theorem) — the
existing stochastic-LLG path, byte-identical to previous package versions for
the same seed. The default `thermostat` of [`run_llg`](@ref).
"""
struct ClassicalThermostat end

"""
    QuantumThermostat()

Semi-quantum colored-noise thermostat (Barker & Bauer, PRB **100**, 140401(R)
(2019)): the thermal-field power spectral density is the classical one times
`θ(x) = x/(eˣ − 1)`, `x = ħ|ω|/kT` — the quantum fluctuation–dissipation
factor **without** the zero-point term — so magnon-mode occupations follow
Bose–Einstein statistics (specific heat → 0 as T → 0) instead of classical
equipartition. Requires a temperature and `kT·dt/ħ ≤ $(_QT_MAX_TAU)` (shrink
`dt` for hot runs; `dt ≤ 0.05·ħ/kT` recommended).

Accuracy notes (see SPEC.md): the damping stays Markovian, so mode occupations
are accurate to ~1% only for `α·(ħω/kT) ≲ 0.03` — keep `α ≲ 0.05` for
quantum-statistics observables. Equilibrium is **not** a Boltzmann ensemble:
fluctuation-formula evaluables are refused by [`equilibrium_stats`](@ref)
(take specific heat from `d⟨E⟩/dT` across runs instead), and the classical MC
cross-checks apply only to [`ClassicalThermostat`](@ref).

The shipped filter (`_QT_S_BIQUADS`, fitted by `dev/fit_qtb_filter.jl`)
matches θ to 0.41% relative over the occupied band — see
`docs/specs/quantum-thermostat.md` for the full certificate. The filter
coefficients are trajectory-defining package constants (`_QT_FILTER_ID`);
checkpoints store them and resume from the stored values verbatim.
"""
struct QuantumThermostat end

const AbstractThermostat = Union{ClassicalThermostat,QuantumThermostat}

_thermostat_string(::ClassicalThermostat)::String = "classical"
_thermostat_string(::QuantumThermostat)::String = "quantum"

# One second-order filter section in direct-form-II-transposed coefficients
# (z-domain `(b0 + b1 z⁻¹ + b2 z⁻²)/(1 + a1 z⁻¹ + a2 z⁻²)`).
struct _Biquad
    b0::Float64
    b1::Float64
    b2::Float64
    a1::Float64
    a2::Float64
end

"""
    ColoredNoiseFilter(sections::Vector{_Biquad}, L::Matrix{Float64})

The discrete-time colored-noise generator of the quantum thermostat: a cascade
of biquad sections applied to the per-step white draws, plus the stationary
square root `L` (`L·Lᵀ` = the stationary state covariance, used ONLY by the
step-0 initialization — never persisted, never touched by the stepping loop).
Built by `_build_quantum_filter(kt, dt)`; user construction is not part of the
API (the coefficients are trajectory-defining and ship as package constants).
"""
struct ColoredNoiseFilter
    sections::Vector{_Biquad}
    L::Matrix{Float64}
    function ColoredNoiseFilter(sections::Vector{_Biquad}, L::Matrix{Float64})
        ns = length(sections)
        1 <= ns <= _QT_MAX_NSECTIONS ||
            throw(ArgumentError("nsections must be in 1:$(_QT_MAX_NSECTIONS); " *
                                "got $ns"))
        for (j, biquad) in enumerate(sections)
            (isfinite(biquad.b0) && isfinite(biquad.b1) && isfinite(biquad.b2) &&
             isfinite(biquad.a1) && isfinite(biquad.a2)) ||
                throw(ArgumentError("section $j has non-finite coefficients"))
            # Jury stability criterion for z² + a1·z + a2 (real coefficients)
            (abs(biquad.a2) < 1 && abs(biquad.a1) < 1 + biquad.a2) ||
                throw(ArgumentError("section $j is unstable (a1 = $(biquad.a1), " *
                                    "a2 = $(biquad.a2))"))
        end
        m = 2 * ns
        size(L) == (m, m) || throw(DimensionMismatch(
            "stationary square root is $(size(L)); expected ($m, $m)"))
        all(isfinite, L) ||
            throw(ArgumentError("stationary square root has non-finite entries"))
        new(copy(sections), copy(L))
    end
end

Base.show(io::IO, f::ColoredNoiseFilter) =
    print(io, "ColoredNoiseFilter($(length(f.sections)) sections)")

# The cascade as a single-input LTI state-space system `s⁺ = A s + B ξ`,
# `y = h·s + d·ξ` with the DF2T state vector (s1_j, s2_j per section, section-
# major) — the algebraic twin of `_qt_cascade!` (the equivalence is a test
# gate). Per section with input u (a linear function of ξ and earlier states):
#   y   = b0·u + s1
#   s1⁺ = (b1 − a1·b0)·u − a1·s1 + s2
#   s2⁺ = (b2 − a2·b0)·u − a2·s1
#
# Parametric in the working precision, with `Float64` the default so every
# existing caller is bit-identical: `_stationary_factor` needs the SAME assembly
# in extended precision, and a second copy of this loop is exactly the drift
# hazard the coupled-site note above exists to prevent.
function _filter_state_space(sections::Vector{_Biquad},
                             ::Type{T} = Float64) where {T<:AbstractFloat}
    m = 2 * length(sections)
    A = zeros(T, m, m)
    B = zeros(T, m)
    crow = zeros(T, m)                   # state row of the current input u_j
    d = one(T)                           # ξ coefficient of the current input
    for (j, biquad) in enumerate(sections)
        p1 = 2 * j - 1
        p2 = 2 * j
        b0, b1, b2 = T(biquad.b0), T(biquad.b1), T(biquad.b2)
        a1, a2 = T(biquad.a1), T(biquad.a2)
        g1 = b1 - a1 * b0
        g2 = b2 - a2 * b0
        for k = 1:m
            A[p1, k] = g1 * crow[k]
            A[p2, k] = g2 * crow[k]
        end
        A[p1, p1] += -a1
        A[p1, p2] += one(T)
        A[p2, p1] += -a2
        B[p1] = g1 * d
        B[p2] = g2 * d
        for k = 1:m                      # u_{j+1} = y_j = b0·u_j + s1_j
            crow[k] *= b0
        end
        crow[p1] += one(T)
        d *= b0
    end
    return A, B, crow, d
end

# Stationary covariance of `s⁺ = A s + B ξ`: the discrete Lyapunov equation
# P = A P Aᵀ + B Bᵀ by direct vec-solve (m ≤ 16). Parametric in the working
# precision — see `_stationary_factor` for why it must not be called at Float64.
function _stationary_cov(A::Matrix{T}, B::Vector{T})::Matrix{T} where {T<:AbstractFloat}
    m = length(B)
    P = reshape((I - kron(A, A)) \ vec(B * B'), m, m)
    return (P + P') / 2
end

# The stationary square root `L` (`L·Lᵀ = P`) of the cascade's state covariance,
# solved in EXTENDED PRECISION and returned as a Cholesky factor.
#
# WHY NOT Float64. The equation and this assembly are correct — solved at 512
# bits the covariance is strictly positive definite at every accepted τ. But the
# vec-system `(I − A⊗A)` is numerically singular there: `cond` reaches 2–5e16,
# past `1/eps`. Two factors multiply into it, and neither is fixable by scaling.
# The bilinear map collapses the poles onto `z = 1` as `1 − ρ(A) = 7.32e-3·τ`
# (measured, exactly proportional), worth ~7e5; the rest is the NON-NORMALITY of
# the DF2T realization, `κ(V) ≈ 454/τ`, which the Kronecker product SQUARES to
# ~3e13. LAPACK balancing was measured and moved `cond` by nothing — a diagonal
# similarity cannot remove non-normality.
#
# What that cost was not a rounding detail. The returned Float64 `P` satisfied
# the Lyapunov equation to 1e-16 while NOT being positive semi-definite, and the
# resulting thermal-noise power `h·P·hᵀ + d²` was wrong by more than 1 % over
# 18.7 % of the accepted range, worst case **+325 %** at τ = 1.70e-4. Confirmed
# end to end on a 216-site run: +127 % = +22.9σ against an independent oracle.
# The error decays as `A^k E A^kᵀ`, half-life ~1e5–5e5 steps, i.e. hundreds of ps
# — far longer than the spin relaxation the bath is supposed to set.
#
# Nothing caught it because the F4 gate compared a stream started from `x = L·ζ`
# against `dot(h, P*h) + d²`: both sides share the same wrong `P`, so it passed
# self-consistently. A reference sharing the core routine is not an oracle
# (`~/Packages/CLAUDE.md`, Testing). The gate that does catch it is the
# Wiener–Khinchin contour integral, built from the stored coefficients alone.
#
# WHY A CHOLESKY FACTOR rather than a clamped eigendecomposition. Solving for a
# factor makes `L·Lᵀ ⪰ 0` STRUCTURAL — there is no negative eigenvalue to clamp,
# so the old `max(·, 0.0)` (whose comment claimed ULP-sized negatives, while the
# real ones reached 5 % of the largest) has nothing to hide. It is also unique
# given positive diagonals, which retires the eigenvector-sign canonicalization
# the previous form needed: LAPACK does not pin its sign choice, and `L` feeds
# the Philox-keyed init, i.e. it sits upstream of every bitwise gate in the
# family. Generic `\`/`cholesky` at BigFloat go through MPFR, which is correctly
# rounded and therefore platform-independent — a shallower dependency than the
# LAPACK path it replaces.
#
# `setprecision` is dynamic-scope global state, which the shared CLAUDE.md warns
# against; the `do` form closes it lexically, and `_QT_LYAP_PRECISION` is a named
# constant rather than a literal so the pinned `L` moves visibly if it changes.
# `precision` is a keyword ONLY so a gate can show the shipped choice has
# headroom (a lower precision must round to the same Float64); production always
# takes the default.
function _stationary_factor(sections::Vector{_Biquad};
                            precision::Int = _QT_LYAP_PRECISION)::Matrix{Float64}
    return setprecision(BigFloat, precision) do
        A, B, _, _ = _filter_state_space(sections, BigFloat)
        P = _stationary_cov(A, B)
        # The identity placeholder (a pure-feedthrough filter) has P ≡ 0, which is
        # PSD but not positive definite, so `cholesky` would throw. Zero is the
        # correct factor and this is the only exactly-singular case the cascade
        # can produce (every fitted section has strictly LHP poles).
        all(iszero, P) && return zeros(Float64, size(P))
        return Float64.(Matrix(cholesky(Symmetric(P)).L))
    end
end

# The discrete filter for one run: the pinned s-domain sections mapped by the
# bilinear transform ŝ = c·(1 − z⁻¹)/(1 + z⁻¹), c = 2/τ, τ = kT·dt/ħ — a pure
# closed-form function of (kT, dt) (decision record Q2/Q4; the discrete PSD is
# exactly θ_fit(x_w), x_w = c·tan(ωΔt/2), so there is no aliasing and the F1
# certificate transfers verbatim). The τ ≤ _QT_MAX_TAU guard has already run.
function _build_quantum_filter(kt::Float64, dt::Float64)::ColoredNoiseFilter
    tau = kt * dt / HBAR_EV_FS
    c = 2.0 / tau
    K = c * c
    sections = Vector{_Biquad}(undef, length(_QT_S_BIQUADS))
    for (j, (β2, β1, β0, α1, α0)) in enumerate(_QT_S_BIQUADS)
        D = K + α1 * c + α0
        sections[j] = _Biquad((β2 * K + β1 * c + β0) / D,
                              2 * (β0 - β2 * K) / D,
                              (β2 * K - β1 * c + β0) / D,
                              2 * (α0 - K) / D,
                              (K - α1 * c + α0) / D)
    end
    return ColoredNoiseFilter(sections, _stationary_factor(sections))
end

# Per-run colored-noise state: the filter and the running DF2T state, one
# column per site, lane rows `lane(c, j, r) = (c − 1)·2·NS + 2·(j − 1) + r`
# (component-major, then section, then the two DF2T slots). Inactive-site
# columns are never drawn and never touched — exact 0.0 (the noise-kernel
# discipline extended to the state).
struct _FilterState
    filter::ColoredNoiseFilter
    x::Matrix{Float64}
end

# Checkpoint (schema v3) coefficient snapshot: column j = (b0, b1, b2, a1, a2)
# of section j — the AUTHORITATIVE record a resume rebuilds the recurrence
# from (coupled site: the reader `_filter_from_coeffs` and the schema doc).
function _filter_coeffs(noise_filter::ColoredNoiseFilter)::Matrix{Float64}
    ns = length(noise_filter.sections)
    coeffs = Matrix{Float64}(undef, 5, ns)
    for (j, biquad) in enumerate(noise_filter.sections)
        coeffs[:, j] .= (biquad.b0, biquad.b1, biquad.b2, biquad.a1, biquad.a2)
    end
    return coeffs
end

# Rebuild the stepping filter from stored coefficients, verbatim. `L` is the
# init-only stationary square root and a resumed run never re-initializes, so
# the rebuilt filter carries a zero placeholder — it must never reach
# `_init_filter_state`.
function _filter_from_coeffs(coeffs::Matrix{Float64})::ColoredNoiseFilter
    size(coeffs, 1) == 5 || error(
        "checkpoint filter coefficients are $(size(coeffs, 1)) × " *
        "$(size(coeffs, 2)); expected 5 rows")
    sections = [_Biquad(coeffs[1, j], coeffs[2, j], coeffs[3, j], coeffs[4, j],
                        coeffs[5, j]) for j = 1:size(coeffs, 2)]
    m = 2 * length(sections)
    return ColoredNoiseFilter(sections, zeros(m, m))
end

# Bitwise filter-state restore — the `_config_verbatim` twin (never rebuild,
# never renormalize: resume must be bit-identical to the uninterrupted run).
function _filter_state_verbatim(m::Matrix{Float64}, nlanes::Int,
                                n::Int)::Matrix{Float64}
    size(m) == (nlanes, n) || error(
        "checkpoint filter state is $(size(m, 1)) × $(size(m, 2)); expected " *
        "$nlanes × $n")
    return copy(m)
end

# Counter for the quantum-thermostat slots (same word layout as `_noise_ctrs`;
# noise.jl's header documents the full map): step ≥ 1 slots 0/1 are the shared
# white inputs (drawn by `_fill_noise_quantum!` via `_noise_ctrs`), step 0
# slots 2 … 2 + 3·NS − 1 are the stationary-init draws.
@inline function _qt_ctr(site::Int, step::Int, slot::UInt32)
    lo = (step % UInt32)
    hi = UInt32((step >>> 32) & 0xFFFF)
    return (UInt32(site), lo, slot, _DOMAIN_SD | hi)
end

# The shared quantum-thermostat resolution of `run_llg` AND `gpu_run_llg`
# (one definition — the validation and the τ bound must never drift between
# the two drivers): returns `nothing` for the classical thermostat, else
# validates and builds the freshly-initialized filter state.
function _resolve_quantum_fstate(thermostat::AbstractThermostat, thermo::Bool,
                                 kt::Float64, dtf::Float64,
                                 H::TiledHamiltonian,
                                 seed_u::UInt64)::Union{Nothing,_FilterState}
    thermostat isa QuantumThermostat || return nothing
    thermo || throw(ArgumentError(
        "the quantum thermostat needs a temperature (pass temperature or kT)"))
    tau = kt * dtf / HBAR_EV_FS
    tau <= _QT_MAX_TAU || throw(ArgumentError(
        "kT·dt/ħ = $(round(tau; sigdigits = 3)) exceeds the quantum " *
        "thermostat's validity bound $(_QT_MAX_TAU) — use dt ≤ " *
        "$(round(_QT_MAX_TAU * HBAR_EV_FS / kt; sigdigits = 3)) fs at this " *
        "temperature (dt ≤ $(round(0.05 * HBAR_EV_FS / kt; sigdigits = 3)) " *
        "fs recommended)"))
    tau >= _QT_MIN_TAU || throw(ArgumentError(
        "kT·dt/ħ = $(round(tau; sigdigits = 3)) is below the quantum " *
        "thermostat's coefficient-conditioning bound " *
        "$(round(_QT_MIN_TAU; sigdigits = 3)) — the discrete sections' DC gain, " *
        "exactly 1 in closed form, is a cancellation of order α₀τ², so below " *
        "this it loses more than $(_QT_DC_TOL) in relative terms. Raise kT, or " *
        "raise dt if the integrator allows it. NOTE the physics is fine here: a " *
        "small τ means a FINE dt, so the Nyquist band (x ≤ π/τ) resolves more of " *
        "θ(x), not less — this is a coefficient-arithmetic limit, not a " *
        "frozen-mode one"))
    return _init_filter_state(_build_quantum_filter(kt, dtf), H, seed_u)
end

# Fresh-run state: every active site starts in the EXACT stationary law of the
# filter (x₀ = L·ζ per component, ζ Philox-keyed at step 0 — no burn-in), so a
# run is a pure function of the seed with no carried state before step 1.
function _init_filter_state(noise_filter::ColoredNoiseFilter, H::TiledHamiltonian,
                            seed::UInt64)::_FilterState
    m = 2 * length(noise_filter.sections)
    n = n_sites(H)
    x = zeros(3 * m, n)
    zeta = Vector{Float64}(undef, 3 * m)
    @inbounds for s = 1:n
        H.site_active[s] || continue
        for b = 0:(3 * m ÷ 2 - 1)
            blk = philox_block(seed, _qt_ctr(s, 0, UInt32(2 + b)))
            n1, n2 = philox_normal2(blk)
            zeta[2 * b + 1] = n1
            zeta[2 * b + 2] = n2
        end
        for c = 1:3
            off = (c - 1) * m
            for i = 1:m
                acc = 0.0
                for k = 1:m
                    acc += noise_filter.L[i, k] * zeta[off + k]
                end
                x[off + i, s] = acc
            end
        end
    end
    return _FilterState(noise_filter, x)
end

# One component's cascade update at site `s` (state rows `off + 1 … off + 2NS`,
# updated in place): the DF2T recurrence, expression order pinned (the GPU
# kernel and the state-space test gate port it literally).
@inline function _qt_cascade!(x::Matrix{Float64}, sections::Vector{_Biquad},
                              s::Int, off::Int, xi::Float64)::Float64
    u = xi
    @inbounds for j in eachindex(sections)
        biquad = sections[j]
        p1 = off + 2 * j - 1
        p2 = off + 2 * j
        out = biquad.b0 * u + x[p1, s]
        x[p1, s] = biquad.b1 * u - biquad.a1 * out + x[p2, s]
        x[p2, s] = biquad.b2 * u - biquad.a2 * out
        u = out
    end
    return u
end

# The quantum-thermostat step fill: the SAME white draws as `_fill_noise!`
# (slots 0/1 — shared realization), pushed through the cascade; `gth[s] =
# σ_s·y_s`. Updated ONCE per step (both integrator stages read the same field —
# the Stratonovich structure, exactly as the classical path).
function _fill_noise_quantum!(gth::Vector{SVector{3,Float64}},
                              H::TiledHamiltonian, sigma::Vector{Float64},
                              x::Matrix{Float64}, noise_filter::ColoredNoiseFilter,
                              seed::UInt64, step::Int)::Nothing
    sections = noise_filter.sections
    m = 2 * length(sections)
    @inbounds for s = 1:n_sites(H)
        if !H.site_active[s]
            gth[s] = zero(SVector{3,Float64})
            continue
        end
        c1, c2 = _noise_ctrs(s, step)
        n1, n2 = philox_normal2(philox_block(seed, c1))
        n3, _ = philox_normal2(philox_block(seed, c2))
        y1 = _qt_cascade!(x, sections, s, 0, n1)
        y2 = _qt_cascade!(x, sections, s, m, n2)
        y3 = _qt_cascade!(x, sections, s, 2 * m, n3)
        gth[s] = sigma[s] * SVector(y1, y2, y3)
    end
    return nothing
end
