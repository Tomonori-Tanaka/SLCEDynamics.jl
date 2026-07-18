# The thermostat selection layer and the semi-quantum colored-noise machinery
# (decision record docs/specs/quantum-thermostat.md, settled spec S1–S18): the
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
# this tag without invalidating old files. "identity-v0" is the Q-M1/Q-M4
# wiring placeholder.
const _QT_FILTER_ID = "identity-v0"

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

!!! warning "Identity placeholder"
    The pinned Barker–Bauer fit constants have NOT landed yet: the current
    filter is the identity, so a `QuantumThermostat()` run produces bitwise
    **classical** white-noise statistics (deliberate — the wiring gate of
    milestone Q-M1, `docs/specs/quantum-thermostat.md`). The type stays
    public-unexported until the physics is real.
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
        for (j, bq) in enumerate(sections)
            (isfinite(bq.b0) && isfinite(bq.b1) && isfinite(bq.b2) &&
             isfinite(bq.a1) && isfinite(bq.a2)) ||
                throw(ArgumentError("section $j has non-finite coefficients"))
            # Jury stability criterion for z² + a1·z + a2 (real coefficients)
            (abs(bq.a2) < 1 && abs(bq.a1) < 1 + bq.a2) ||
                throw(ArgumentError("section $j is unstable (a1 = $(bq.a1), " *
                                    "a2 = $(bq.a2))"))
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
function _filter_state_space(sections::Vector{_Biquad})
    m = 2 * length(sections)
    A = zeros(m, m)
    B = zeros(m)
    crow = zeros(m)                      # state row of the current input u_j
    d = 1.0                              # ξ coefficient of the current input
    for (j, bq) in enumerate(sections)
        p1 = 2 * j - 1
        p2 = 2 * j
        g1 = bq.b1 - bq.a1 * bq.b0
        g2 = bq.b2 - bq.a2 * bq.b0
        for k = 1:m
            A[p1, k] = g1 * crow[k]
            A[p2, k] = g2 * crow[k]
        end
        A[p1, p1] += -bq.a1
        A[p1, p2] += 1.0
        A[p2, p1] += -bq.a2
        B[p1] = g1 * d
        B[p2] = g2 * d
        for k = 1:m                      # u_{j+1} = y_j = b0·u_j + s1_j
            crow[k] *= bq.b0
        end
        crow[p1] += 1.0
        d *= bq.b0
    end
    return A, B, crow, d
end

# Stationary covariance of `s⁺ = A s + B ξ`: the discrete Lyapunov equation
# P = A P Aᵀ + B Bᵀ by direct vec-solve (m ≤ 16 — exact and deterministic).
function _stationary_cov(A::Matrix{Float64}, B::Vector{Float64})::Matrix{Float64}
    m = length(B)
    P = reshape((I - kron(A, A)) \ vec(B * B'), m, m)
    return (P + P') / 2
end

# A square root L with L·Lᵀ = P via the clamped symmetric eigendecomposition —
# NOT a plain Cholesky, which throws on the ULP-negative eigenvalues of
# near-singular P (small τ) and on the exactly zero P of a pure-feedthrough
# filter (the identity placeholder). Eigenvector signs are canonicalized
# (largest-|component| entry made positive) so L is a deterministic function
# of P — LAPACK's sign choice is not otherwise pinned, and L feeds the
# Philox-keyed stationary init (bit-reproducibility surface).
function _stationary_sqrt(P::Matrix{Float64})::Matrix{Float64}
    e = eigen(Symmetric(P))
    V = e.vectors
    for k = 1:size(V, 2)
        col = @view V[:, k]
        i = argmax(abs.(col))
        col[i] < 0 && (col .*= -1)
    end
    return V * Diagonal(sqrt.(max.(e.values, 0.0)))
end

# The discrete filter for one run, a pure closed-form function of (kT, dt).
#
# MILESTONE PLACEHOLDER (wiring tier): the identity section — a flat unit PSD,
# so the quantum path is bitwise the classical one (the wiring gate in
# test_quantum_thermostat.jl). The pinned Barker–Bauer fit constants and the
# bilinear (kT, dt) mapping replace this body; every seam (τ guard, state
# layout, counters, checkpoint, GPU) is already final.
function _build_quantum_filter(kt::Float64, dt::Float64)::ColoredNoiseFilter
    sections = [_Biquad(1.0, 0.0, 0.0, 0.0, 0.0)]
    A, B, _, _ = _filter_state_space(sections)
    return ColoredNoiseFilter(sections, _stationary_sqrt(_stationary_cov(A, B)))
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
function _filter_coeffs(filt::ColoredNoiseFilter)::Matrix{Float64}
    ns = length(filt.sections)
    coeffs = Matrix{Float64}(undef, 5, ns)
    for (j, bq) in enumerate(filt.sections)
        coeffs[:, j] .= (bq.b0, bq.b1, bq.b2, bq.a1, bq.a2)
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

# Fresh-run state: every active site starts in the EXACT stationary law of the
# filter (x₀ = L·ζ per component, ζ Philox-keyed at step 0 — no burn-in), so a
# run is a pure function of the seed with no carried state before step 1.
function _init_filter_state(filt::ColoredNoiseFilter, H::TiledHamiltonian,
                            seed::UInt64)::_FilterState
    m = 2 * length(filt.sections)
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
                    acc += filt.L[i, k] * zeta[off + k]
                end
                x[off + i, s] = acc
            end
        end
    end
    return _FilterState(filt, x)
end

# One component's cascade update at site `s` (state rows `off + 1 … off + 2NS`,
# updated in place): the DF2T recurrence, expression order pinned (the GPU
# kernel and the state-space test gate port it literally).
@inline function _qt_cascade!(x::Matrix{Float64}, sections::Vector{_Biquad},
                              s::Int, off::Int, xi::Float64)::Float64
    u = xi
    @inbounds for j in eachindex(sections)
        bq = sections[j]
        p1 = off + 2 * j - 1
        p2 = off + 2 * j
        out = bq.b0 * u + x[p1, s]
        x[p1, s] = bq.b1 * u - bq.a1 * out + x[p2, s]
        x[p2, s] = bq.b2 * u - bq.a2 * out
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
                              x::Matrix{Float64}, filt::ColoredNoiseFilter,
                              seed::UInt64, step::Int)::Nothing
    sections = filt.sections
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
