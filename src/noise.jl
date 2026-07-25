# The stochastic-LLG thermal field (SPEC.md "Stochastic LLG"): a per-site,
# per-step Gaussian gradient contribution
#
#     G_th,i = σ_i · ξ_i,   σ_i = √(2 α_i kB T ħ magmom_i / (g_i Δt))   [eV]
#
# with ξ_i ~ N(0, I₃), derived from the fluctuation–dissipation constant
# D = α kB T/(γ·magmom·μ_B) — NO (1+α²) factor in this parametrization (the noise
# rides the same (1+α²)-prefactored equation (★) as the deterministic field; the
# runtime authority is the α-independence Boltzmann gate in test_thermostat.jl).
# The same draw is used by BOTH integrator stages of one step — that is what makes
# the Heun structure converge to the Stratonovich solution.
#
# Draws are keyed philox4x32-10 (SLCEMonteCarlo's public facade — Random123
# known-answer-gated): a draw is a pure function of (seed, site, step), so
# trajectories are bit-reproducible for any ntasks and need no stored RNG state.
#
# Counter layout (the upstream contract requires a NONZERO word-4 domain tag —
# MC streams use 0):
#
#     ctr = (site, step_lo32, slot, _DOMAIN_SD | step_hi16)
#
# Slot map (thermostat.jl's `_qt_ctr` shares the word layout — coupled site):
#   step ≥ 1, slots 0/1 — the white triple, drawn by BOTH thermostats (the
#     quantum cascade filters these same draws — shared realization); the
#     fourth normal of slot 1 is discarded (reserved).
#   step 0, slots 2 … 2 + 3·NS − 1 — the quantum thermostat's stationary-init
#     draws (step 0 is never drawn by the stepping loop, which starts at 1).
# Step capacity 2^48.

const _DOMAIN_SD = 0x53440000        # "SD" in the high half of counter word 4

@inline function _noise_ctrs(site::Int, step::Int)
    lo = (step % UInt32)
    hi = UInt32((step >>> 32) & 0xFFFF)
    w4 = _DOMAIN_SD | hi
    s = UInt32(site)
    return (s, lo, 0x00000000, w4), (s, lo, 0x00000001, w4)
end

# Per-site noise amplitude [eV] for one step size; exact zero on inactive sites
# (no thermostat on frozen spins).
function _sigma_noise(prob::LLGProblem, kt::Float64, dt::Float64)::Vector{Float64}
    n = n_sites(prob.H)
    sigma = zeros(n)
    for s = 1:n
        prob.H.site_active[s] || continue
        sigma[s] = sqrt(2 * prob.alpha[s] * kt * HBAR_EV_FS * prob.magmom[s] /
                        (prob.g[s] * dt))
    end
    return sigma
end

# Fill the per-step thermal-gradient buffer: pure function of (seed, site, step).
function _fill_noise!(gth::Vector{SVector{3,Float64}}, H::TiledHamiltonian,
                      sigma::Vector{Float64}, seed::UInt64, step::Int)::Nothing
    @inbounds for s = 1:n_sites(H)
        if !H.site_active[s]
            gth[s] = zero(SVector{3,Float64})
            continue
        end
        c1, c2 = _noise_ctrs(s, step)
        n1, n2 = philox_normal2(philox_block(seed, c1))
        n3, _ = philox_normal2(philox_block(seed, c2))
        gth[s] = sigma[s] * SVector(n1, n2, n3)
    end
    return nothing
end
