# Fixed-step deterministic integrators. Both are Heun-structured (two field
# evaluations per step — the dominant cost, ≈ 2 Metropolis sweeps) and RNG-free,
# so trajectories are bit-reproducible for any `ntasks` (the only threaded piece
# is `energy_gradient!`, itself bit-identical for any task count).
#
# The rotation-vector form of the LLG: de/dt = ω × e with
#
#     ω_s = −p_s · (G_s + α_s e_s × G_s),   G_s = G_SCE,s + G_Zeeman,s
#
# (verified: ω×e = p[e×G + α e×(e×G)], the working equation). The radial
# component of ω is irrelevant to the continuous flow (ω ∥ e contributes ω×e = 0);
# `G_SCE` arrives tangent-projected from `energy_gradient!` while the constant
# Zeeman gradient is deliberately NOT projected — for a pure external field the
# rotation axis is then exactly `B_ext`, and Depondt–Mertens reproduces uniform
# Larmor precession to rounding at any step size.

"""
    DepondtMertens()

Rotation-based Heun integrator (Depondt & Mertens, J. Phys.: Condens. Matter 21,
336005 (2009)): predictor and corrector each advance every spin by a rigid
Rodrigues rotation about its local rotation vector `ω_s`, so `|e_s| = 1` is
preserved to rounding with no renormalization step. The corrector rotates the
*original* spin about the stage-averaged `(ω¹ + ω²)/2`. Weak/deterministic order
2 in `dt`; exact for a constant rotation vector (pure Zeeman precession).
The recommended default.
"""
struct DepondtMertens end

"""
    HeunProjected()

Classical explicit Heun with per-stage renormalization (the standard reference
scheme of the stochastic-LLG literature, e.g. García-Palacios & Lázaro, PRB 58,
14937 (1998)): Euler predictor along `de/dt = ω × e`, trapezoidal corrector, each
followed by `e → e/|e|`. Deterministic order 2. Kept as an independent
cross-check of [`DepondtMertens`](@ref) — two structurally different integrators
agreeing to O(dt²) is a strong implementation gate.
"""
struct HeunProjected end

const AbstractIntegrator = Union{DepondtMertens,HeunProjected}

# Per-run work buffers (never shared across concurrent runs).
struct _LLGScratch
    G::Vector{SVector{3,Float64}}       # all-site gradient (both stages, in place)
    omega1::Vector{SVector{3,Float64}}  # stage-1 rotation vectors
    epred::SpinConfig                   # predictor configuration
end
_LLGScratch(n::Int) = _LLGScratch(Vector{SVector{3,Float64}}(undef, n),
                                  Vector{SVector{3,Float64}}(undef, n),
                                  SpinConfig(undef, n))

# ω_s at spin `e` given the SCE gradient `gsce` (site-active callers only).
@inline function _omega(prob::LLGProblem, s::Int, e::SVector{3,Float64},
                        gsce::SVector{3,Float64})::SVector{3,Float64}
    gt = gsce + prob.gzee[s]
    return -prob.pref[s] * (gt + prob.alpha[s] * cross(e, gt))
end

# Rodrigues rotation of `v` by the axis-angle vector `w` (angle |w|, axis w/|w|).
# `w == 0` returns `v` unchanged — the divide-by-zero guard for a genuinely zero
# rotation vector on an active site (frozen sites never reach this call).
@inline function _rotate(v::SVector{3,Float64},
                         w::SVector{3,Float64})::SVector{3,Float64}
    θ = norm(w)
    θ == 0.0 && return v
    n = w / θ
    sθ, cθ = sincos(θ)
    return v * cθ + cross(n, v) * sθ + n * (dot(n, v) * (1 - cθ))
end

# One Depondt–Mertens step, in place. Inactive sites are skipped bitwise at both
# stages (their `epred` entry is still filled so stage 2 sees a full config).
function _step!(::DepondtMertens, config::SpinConfig, prob::LLGProblem,
                dt::Float64, sc::_LLGScratch, ntasks::Int)::Nothing
    H = prob.H
    energy_gradient!(sc.G, H, config; ntasks = ntasks)
    @inbounds for s = 1:n_sites(H)
        if !H.site_active[s]
            sc.epred[s] = config[s]
            continue
        end
        ω1 = _omega(prob, s, config[s], sc.G[s])
        sc.omega1[s] = ω1
        sc.epred[s] = _rotate(config[s], ω1 * dt)
    end
    energy_gradient!(sc.G, H, sc.epred; ntasks = ntasks)
    @inbounds for s = 1:n_sites(H)
        H.site_active[s] || continue
        ω2 = _omega(prob, s, sc.epred[s], sc.G[s])
        config[s] = _rotate(config[s], (sc.omega1[s] + ω2) * (dt / 2))
    end
    return nothing
end

# One projected-Heun step, in place.
function _step!(::HeunProjected, config::SpinConfig, prob::LLGProblem,
                dt::Float64, sc::_LLGScratch, ntasks::Int)::Nothing
    H = prob.H
    energy_gradient!(sc.G, H, config; ntasks = ntasks)
    @inbounds for s = 1:n_sites(H)
        if !H.site_active[s]
            sc.epred[s] = config[s]
            continue
        end
        ω1 = _omega(prob, s, config[s], sc.G[s])
        sc.omega1[s] = ω1
        ep = config[s] + dt * cross(ω1, config[s])
        sc.epred[s] = ep / norm(ep)
    end
    energy_gradient!(sc.G, H, sc.epred; ntasks = ntasks)
    @inbounds for s = 1:n_sites(H)
        H.site_active[s] || continue
        ω2 = _omega(prob, s, sc.epred[s], sc.G[s])
        e = config[s] + (dt / 2) * (cross(sc.omega1[s], config[s]) +
                                    cross(ω2, sc.epred[s]))
        config[s] = e / norm(e)
    end
    return nothing
end
