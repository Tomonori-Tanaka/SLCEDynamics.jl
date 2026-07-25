# The immutable per-run problem definition: Hamiltonian + per-site material
# parameters, with the LLG prefactor and the (constant) Zeeman gradient resolved
# once at construction.

"""
    LLGProblem(H::TiledHamiltonian; magmom, alpha = 0.0, g = 2.0,
               b_ext = (0.0, 0.0, 0.0)) -> LLGProblem

Definition of one LLG dynamics run on the SCE Hamiltonian `H`.

- `magmom` (required): moment magnitudes in μ_B — a scalar (uniform), a vector of
  length `n_sites(H)`, or a vector with one entry per training-cell atom (tiled to
  the supercell via `site_atom`). Must be positive and finite on every **active**
  site; inactive sites (no adjacent SCE instance — outside the fitted magnetic
  subsystem) are frozen entirely and their entries are ignored.
- `alpha`: Gilbert damping, scalar or per-site/per-atom vector, `≥ 0`.
- `g`: gyromagnetic g-factor, scalar or per-site/per-atom vector, `> 0`. With
  non-uniform `g` the total moment of a rotation-invariant model is *not*
  conserved at `α = 0` (each sublattice precesses at its own rate) — expected
  physics, not an integrator defect.
- `b_ext`: uniform external field in **tesla** (3-vector). Enters as the Zeeman
  energy `E_Z = −Σ_i magmom_i·μ_B·(e_i·b_ext)` over active sites.

The equation of motion (unit vectors, `G_i = ∂E/∂e_i` in model energy units):

    de_i/dt = p_i · [e_i × G_i + α_i e_i × (e_i × G_i)],
    p_i = g_i / (ħ · magmom_i · (1 + α_i²))    [1/(eV·fs)]

Inactive sites have `p_i = 0` and are additionally skipped bitwise by the
integrators (mirroring `SLCEMonteCarlo`'s frozen-spin convention).
"""
struct LLGProblem
    H::TiledHamiltonian
    magmom::Vector{Float64}          # per site [μ_B]
    alpha::Vector{Float64}           # per site
    g::Vector{Float64}               # per site
    b_ext::SVector{3,Float64}        # [T]
    pref::Vector{Float64}            # g/(ħ·magmom·(1+α²)) [1/(eV·fs)]; 0 on inactive
    gzee::Vector{SVector{3,Float64}} # Zeeman gradient −magmom·μ_B·b_ext; 0 on inactive

    function LLGProblem(H::TiledHamiltonian; magmom, alpha = 0.0, g = 2.0,
                        b_ext = (0.0, 0.0, 0.0))
        n = n_sites(H)
        mm = _per_site(H, magmom, "magmom")
        al = _per_site(H, alpha, "alpha")
        gf = _per_site(H, g, "g")
        length(b_ext) == 3 || throw(ArgumentError(
            "b_ext must be a 3-vector [T]; got length $(length(b_ext))"))
        b = SVector{3,Float64}(b_ext)
        all(isfinite, b) || throw(ArgumentError("b_ext must be finite; got $b"))
        pref = zeros(n)
        gzee = fill(zero(SVector{3,Float64}), n)
        for s = 1:n
            H.site_active[s] || continue
            (isfinite(mm[s]) && mm[s] > 0) || throw(ArgumentError(
                "magmom must be positive and finite on active sites; " *
                "got $(mm[s]) at site $s"))
            (isfinite(al[s]) && al[s] >= 0) || throw(ArgumentError(
                "alpha must be ≥ 0 and finite; got $(al[s]) at site $s"))
            (isfinite(gf[s]) && gf[s] > 0) || throw(ArgumentError(
                "g must be positive and finite; got $(gf[s]) at site $s"))
            pref[s] = gf[s] / (HBAR_EV_FS * mm[s] * (1 + al[s]^2))
            gzee[s] = -mm[s] * MU_B_EV_T * b
        end
        return new(H, mm, al, gf, b, pref, gzee)
    end
end

# Resolve a scalar / per-atom / per-site parameter to a dense per-site vector.
# Per-atom vectors are tiled through `site_atom` (every supercell copy of one
# training-cell atom shares its value; assumes the upstream contract that atom
# indices are contiguous `1:natoms` — a coupled site with SLCEMonteCarlo's
# `site_atom`). When the supercell is trivial the two vector lengths coincide
# and so do the interpretations.
function _per_site(H::TiledHamiltonian, x, name::String)::Vector{Float64}
    n = n_sites(H)
    x isa Real && return fill(Float64(x), n)
    x isa AbstractVector{<:Real} || throw(ArgumentError(
        "$name must be a Real or a vector of Reals; got $(typeof(x))"))
    length(x) == n && return Vector{Float64}(x)
    natoms = maximum(site_atom(H, s) for s = 1:n)
    length(x) == natoms &&
        return Float64[x[site_atom(H, s)] for s = 1:n]
    throw(ArgumentError("$name has length $(length(x)); expected a scalar, " *
                        "$natoms (per training-cell atom), or $n (per site)"))
end

function Base.show(io::IO, prob::LLGProblem)
    n = n_sites(prob.H)
    # report parameter ranges over ACTIVE sites only — inactive entries are
    # unvalidated placeholders (deliberately ignored by the constructor)
    act = findall(s -> prob.H.site_active[s], 1:n)
    if isempty(act)
        print(io, "LLGProblem($(n) sites, none active)")
        return
    end
    mm = extrema(prob.magmom[act])
    al = extrema(prob.alpha[act])
    print(io, "LLGProblem($(n) sites, magmom ∈ [$(mm[1]), $(mm[2])] μ_B, ",
          "α ∈ [$(al[1]), $(al[2])], |B_ext| = $(norm(prob.b_ext)) T)")
end

"""
    SLCEMonteCarlo.total_energy(prob::LLGProblem, config::SpinConfig) -> Float64

The dynamical energy of `config`: the SCE energy (`total_energy(prob.H, config)`,
intercept excluded) plus the Zeeman energy `−Σ_i magmom_i·μ_B·(e_i·b_ext)` over
active sites. This is the conserved quantity at `α = 0` and the monotonically
decreasing one at `α > 0` (with `T = 0`).
"""
SLCEMonteCarlo.total_energy(prob::LLGProblem, config::SpinConfig)::Float64 =
    SLCEMonteCarlo.total_energy(prob.H, config) + _zeeman_energy(prob, config)

# The Zeeman part alone; `gzee = −magmom·μ_B·b_ext = ∂E_Z/∂e`, so `E_Z = Σ gzee·e`
# over active sites.
function _zeeman_energy(prob::LLGProblem, config::SpinConfig)::Float64
    E = 0.0
    @inbounds for s = 1:n_sites(prob.H)
        prob.H.site_active[s] || continue
        E += dot(prob.gzee[s], config[s])
    end
    return E
end
