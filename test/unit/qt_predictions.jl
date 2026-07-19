# Q-M4 gate predictions: exact linear-response occupation of a damped precession
# mode under the (1+α²)-parametrized sLLG thermal field with an arbitrary
# thermostat PSD factor P(ω) on the Nyquist band.
#
# Derivation (package conventions, SPEC.md):
#   de/dt = ω×e, ω = −p(G + α e×G), p = g/(ħ μ (1+α²)), G_Zee = −b ẑ, b = μ·μ_B·B.
#   Linearizing about e = +ẑ (m = transverse part, m₊ = m_x + i m_y):
#       dm₊/dt = (i − α)·p·(g₊ − c·m₊),   c = ẑ-part of G, g₊ = transverse G
#   Single Zeeman spin (c = −b): pole  m₊ ∝ e^{(iω₀ − Γ)t},
#       ω₀ = p·b = ω̃/(1+α²),  Γ = α·p·b = α·ω̃/(1+α²),  ω̃ = g·μ_B·B/ħ.
#   Thermal field G_th = σξ, σ²Δt = 2·α·kT·ħ·μ/g  ⇒  the transverse force
#   f₊ = (i−α)·p·σ·ξ₊ has two-sided PSD S_f(ω) = 2(1+α²)p²σ²Δt·P(ω), so
#       ⟨|m₊|²⟩ = ∫ S_f(ω)/((ω−ω₀)² + Γ²) dω/2π,
#   and with E − E₀ = b(1 − e_z) = (b/2)|m₊|² + O(m⁴) all constants collapse:
#       ⟨E⟩ − E₀ = (kT·Γ/π) ∫_band P(ω)/((ω−ω₀)² + Γ²) dω
#   (band = [−π/Δt, π/Δt]; P ≡ 1 on an infinite band gives exactly kT).
#
# Evaluated with the tan substitution ω = ω_m + Γ·tan u (exact Lorentzian
# weight): ⟨E⟩−E₀ = (kT/π) ∫ P(ω_m + Γ tan u) du, split at ω = 0 where the
# quantum P(ω) = θ(ħ|ω|/kT) has its |ω| kink.


# θ(x) = x/(eˣ−1): the zero-point-free quantum FDT factor. θ(0) = 1.
_qt_theta(x::Float64)::Float64 = x == 0.0 ? 1.0 : x / expm1(x)

# Composite Simpson of f on [a, b] with n (odd, ≥ 3) points.
function _qt_simpson(f, a::Float64, b::Float64, n::Int)::Float64
    @assert isodd(n) && n >= 3
    h = (b - a) / (n - 1)
    s = f(a) + f(b)
    for i = 2:2:(n - 1)
        s += 4 * f(a + (i - 1) * h)
    end
    for i = 3:2:(n - 2)
        s += 2 * f(a + (i - 1) * h)
    end
    return s * h / 3
end

"""
    _qt_mode_energy(kt, dt, omega_m, gamma_m, Pfun; npts = 40_001) -> Float64

Stationary ⟨E⟩ − E₀ [eV] of ONE damped circular precession mode (pole
`ω_m − iΓ_m`, both [rad/fs]) under the sLLG thermal field whose PSD is the
classical white level times `Pfun(ω)` on the Nyquist band `|ω| ≤ π/dt`:

    ⟨E⟩ − E₀ = (kT·Γ_m/π) ∫_{−π/dt}^{π/dt} Pfun(ω) dω / ((ω − ω_m)² + Γ_m²)

`Pfun = ω -> θ(ħ|ω|/kT)` is the ideal quantum target; the shipped filter's
closed-form discrete PSD is `_qt_filter_psd(coeffs, dt)`. A zero mode
(`omega_m == 0 && gamma_m == 0`) carries no energy and returns 0.

Discrete-time scope (measured, α = 0.5, ω̃dt ≤ 0.091): the SIMULATION tracks the
Boltzmann value kT in the classical case — NOT this integral's band-truncated
P = 1 value (kT·(1 − 2Γdt/π² + …), rejected at 5σ) — because the discrete-map
response compensates the Lorentzian Nyquist-tail loss (exact-map correction is
only O((ω_m dt)²/12)). Therefore: classical anchors must compare against kT
exactly; quantum predictions may use this integral as-is, since with P = θ the
Lorentzian tail is exponentially cut long before the band edge (band
sensitivity < 1e-13 at the gate parameters). Residual discrete/integrator
systematic: |bias| < 0.4 % for ω_m·dt ≤ 0.09, α ≤ 0.5 (dt-halving runs).
"""
function _qt_mode_energy(kt::Float64, dt::Float64, omega_m::Float64,
                         gamma_m::Float64, Pfun; npts::Int = 40_001)::Float64
    omega_m == 0.0 && gamma_m == 0.0 && return 0.0
    gamma_m > 0.0 || throw(ArgumentError("gamma_m must be > 0; got $gamma_m"))
    wn = pi / dt
    ulo = atan(-(wn + omega_m) / gamma_m)
    uhi = atan((wn - omega_m) / gamma_m)
    umid = atan(-omega_m / gamma_m)          # ω = 0 (kink of θ(|ω|))
    f(u) = Pfun(omega_m + gamma_m * tan(u))
    return (kt / pi) * (_qt_simpson(f, ulo, umid, npts) +
                        _qt_simpson(f, umid, uhi, npts))
end

"""
    _qt_predict_occupation(kt, dt, omega_t, alpha, Pfun; npts = 40_001) -> Float64

Predicted stationary ⟨E⟩ − E₀ [eV] of a single active spin in a field B ∥ ẑ
(Zeeman only) under the package sLLG: `omega_t = g·μ_B·B/ħ` [rad/fs] is the
UNDAMPED Larmor rate (`_larmor_omega`), the actual pole is
`ω₀ = omega_t/(1+α²)`, `Γ = α·omega_t/(1+α²)`. Exact relations:

    ⟨1 − e_z⟩ = (⟨E⟩ − E₀)/b,  b = magmom·μ_B·B = (magmom/g)·ħ·omega_t  [eV]
    n_eff     = (⟨E⟩ − E₀)/(ħ·ω₀)    (→ n_BE(ħω₀/kT) as α → 0 with P = θ)

Valid for b ≫ kT_eff (linear spin-wave regime; the classical anharmonic
correction is 1 − 2y/(e^{2y}−1), y = b/kT — exponentially small in y).
"""
function _qt_predict_occupation(kt::Float64, dt::Float64, omega_t::Float64,
                                alpha::Float64, Pfun;
                                npts::Int = 40_001)::Float64
    omega0 = omega_t / (1 + alpha^2)
    gamma = alpha * omega_t / (1 + alpha^2)
    return _qt_mode_energy(kt, dt, omega0, gamma, Pfun; npts = npts)
end

# The ideal quantum target PSD ω ↦ θ(ħ|ω|/kT).
_qt_theta_psd(kt::Float64) = (w::Float64) -> _qt_theta(SD.HBAR_EV_FS * abs(w) / kt)

# Closed-form discrete PSD of a biquad cascade (coeffs 5 × NS, column j =
# (b0, b1, b2, a1, a2) — the checkpoint/`_filter_coeffs` layout) at z = e^{iωΔt}.
function _qt_filter_psd(coeffs::Matrix{Float64}, dt::Float64)
    size(coeffs, 1) == 5 || throw(ArgumentError("coeffs must be 5 × NS"))
    return function (w::Float64)
        zi = cis(-w * dt)                # z⁻¹
        zi2 = zi * zi
        acc = 1.0
        for j = 1:size(coeffs, 2)
            num = coeffs[1, j] + coeffs[2, j] * zi + coeffs[3, j] * zi2
            den = 1.0 + coeffs[4, j] * zi + coeffs[5, j] * zi2
            acc *= abs2(num) / abs2(den)
        end
        return acc
    end
end

"""
    _qt_dimer_modes(J, bz, magmom, g, alpha) -> NamedTuple

Linear mode frequencies/linewidths [rad/fs] of the ferro dimer `E = J e₁·e₂`
(J < 0) + field `bz` [T] ∥ ẑ about parallel alignment (both spins identical
`magmom`, `g`, `alpha`):

    p   = g/(ħ·magmom·(1+α²)),  b = magmom·μ_B·bz  [eV]
    uniform:  ω_u = p·b            (exchange cancels; Larmor/(1+α²))
    optical:  ω_o = p·(b + 2|J|)
    Γ_m = α·ω_m  for both (damping is per-site and both modes are
    site-symmetric/antisymmetric combinations of identical sites).

Quadratic energy: E − E₀ = (b/2)|u|² + ((b+2|J|)/2)|w|², u/w = (m₁ ± m₂)/√2,
so ⟨1 − e₁·e₂⟩ = ⟨|w|²⟩ and Σᵢ⟨1 − e_z,i⟩ = (⟨|u|²⟩ + ⟨|w|²⟩)/2.
"""
function _qt_dimer_modes(J::Float64, bz::Float64, magmom::Float64, g::Float64,
                         alpha::Float64)
    J < 0 || throw(ArgumentError("ferro dimer needs J < 0; got $J"))
    p = g / (SD.HBAR_EV_FS * magmom * (1 + alpha^2))
    b = magmom * SD.MU_B_EV_T * bz
    omega_u = p * b
    omega_o = p * (b + 2 * abs(J))
    return (; omega_u, omega_o, gamma_u = alpha * omega_u,
            gamma_o = alpha * omega_o, b_u = b, b_o = b + 2 * abs(J))
end

"""
    _qt_predict_dimer(kt, dt, J, bz, magmom, g, alpha, Pfun) -> NamedTuple

⟨E⟩ − E₀ [eV] of the thermostatted dimer = Σ_modes one-mode integrals (site
noises are independent ⇒ the orthonormal mode transform (m₁ ± m₂)/√2 gives two
independent circular modes with the SAME per-mode noise PSD as a single site).
Also returns the mode-resolved observables
`⟨1 − e₁·e₂⟩ = ⟨|w|²⟩ = 2·E_o/b_o` and `Σᵢ⟨1 − e_z,i⟩ = E_u/b_u + E_o/b_o`.
At `bz = 0` the uniform mode is the free rotation zero mode (no energy).
"""
function _qt_predict_dimer(kt::Float64, dt::Float64, J::Float64, bz::Float64,
                           magmom::Float64, g::Float64, alpha::Float64, Pfun;
                           npts::Int = 40_001)
    md = _qt_dimer_modes(J, bz, magmom, g, alpha)
    Eu = _qt_mode_energy(kt, dt, md.omega_u, md.gamma_u, Pfun; npts = npts)
    Eo = _qt_mode_energy(kt, dt, md.omega_o, md.gamma_o, Pfun; npts = npts)
    return (; E = Eu + Eo, Eu, Eo, one_m_e12 = 2 * Eo / md.b_o,
            sum_one_m_ez = (md.omega_u == 0.0 ? 0.0 : Eu / md.b_u) + Eo / md.b_o,
            modes = md)
end

# Exact CLASSICAL Boltzmann ⟨E⟩ − E₀ for E = −b·e_z on the unit sphere (also the
# exchange-only dimer with b → |J|): kT·(1 − 2y/(e^{2y} − 1)), y = b/kT — the
# anharmonicity reference for gate-tolerance budgeting.
_qt_classical_exact(kt::Float64, b::Float64)::Float64 =
    kt * (1 - 2 * (b / kt) / expm1(2 * b / kt))

"""
    _qt_predict_ring(kt, dt, J, bz, magmom, g, alpha, N, Pfun) -> NamedTuple

Per-mode occupations of the ferro `N`-ring `E = J Σ_bonds e_i·e_{i+1}` (J < 0)
+ field `bz` [T] ∥ ẑ about all-up. Linearizing (m₊ⱼ = e_x,j + i·e_y,j) and
Fourier-transforming with the unitary `c_m = N^{-1/2} Σ_j e^{-i·2πmj/N} m₊ⱼ`
gives N INDEPENDENT circular modes (site noises are i.i.d. ⇒ the unitary
transform preserves the per-mode PSD), each exactly the single-mode problem of
`_qt_mode_energy`:

    E − E₀ = Σ_m (b_m/2)·|c_m|²,   b_m = b + 2|J|·(1 − cos(2πm/N))  [eV]
    ω_m = p·b_m,  Γ_m = α·ω_m,     p = g/(ħ·magmom·(1+α²))

Returns `b_m`, `E_m` (⟨E⟩ − E₀ per mode), and `I_m = ⟨|c_m|²⟩ = 2E_m/b_m` —
the ω-INTEGRATED inelastic `S⁺⁻(q_m)` bin sum of `structure_factor` divided by
`M·Δt_meas` (its `F₊(q_m)` with all sites active is exactly `c_m`), and equally
the equal-time `|F₊(q_m)|²` observable. Mode m and N − m are degenerate.
"""
function _qt_predict_ring(kt::Float64, dt::Float64, J::Float64, bz::Float64,
                          magmom::Float64, g::Float64, alpha::Float64, N::Int,
                          Pfun; npts::Int = 40_001)
    J < 0 || throw(ArgumentError("ferro ring needs J < 0; got $J"))
    p = g / (SD.HBAR_EV_FS * magmom * (1 + alpha^2))
    b = magmom * SD.MU_B_EV_T * bz
    b_m = [b + 2 * abs(J) * (1 - cospi(2 * m / N)) for m = 0:(N - 1)]
    E_m = [bm == 0.0 ? 0.0 :
           _qt_mode_energy(kt, dt, p * bm, alpha * p * bm, Pfun; npts = npts)
           for bm in b_m]
    return (; b_m, E_m, I_m = [bm == 0.0 ? 0.0 : 2 * Em / bm
                               for (Em, bm) in zip(E_m, b_m)])
end
