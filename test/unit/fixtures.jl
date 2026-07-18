# Shared fixtures for the unit suite, mirroring SCEMonteCarlo's test fixtures.
# `SD` aliases the package so internal names resolve as `SD._name`.

using SCESpinDynamics
using SCEMonteCarlo
using SCEFitting
using Spglib: Spglib          # activates SCEFitting's SpglibBackend extension
using LinearAlgebra
using Random
using StaticArrays
using Test

const SD = SCESpinDynamics
const MC = SCEMonteCarlo

# A ferromagnetic Heisenberg dimer (the SCEMonteCarlo fixture): 4 atoms in a
# column, but only the first SALC — the atom-1–2 bond — carries a coefficient, so
# atoms 3–4 are inactive (frozen by the dynamics). E = J e₁·e₂ exactly.
function _dimer_crystal()
    lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
    return Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
end

function _dimer_model(J::Float64 = -0.02)
    b = SCEBasis(_dimer_crystal(), BasisSpec(; nbody = 2, cutoff = 2.6,
                                             lmax = [1], isotropy = true))
    return SCEPredictor(b, 0.0, vcat([J], zeros(n_salcs(b) - 1)))
end

# The pair coupling J of the dimer (E = J e₁·e₂), read off tiled energies.
function _dimer_J(H::MC.TiledHamiltonian)
    up = SVector(0.0, 0.0, 1.0)
    aligned = MC.SpinConfig([up for _ = 1:H.n_sites])
    anti = copy(aligned)
    anti[1] = -up
    return (total_energy(H, aligned) - total_energy(H, anti)) / 2
end

# The 4-atom Heisenberg RING on the dimer crystal: the same NN-pair basis, but
# every bond carries J (bonds 1–2, 2–3, 3–4 and the periodic 4–1 image, all
# 2.5 Å < cutoff 2.6) — the S(q,ω) dispersion fixture. All sites active.
function _ring_model(J::Float64 = -0.02)
    b = SCEBasis(_dimer_crystal(), BasisSpec(; nbody = 2, cutoff = 2.6,
                                             lmax = [1], isotropy = true))
    @assert n_salcs(b) == 4          # exactly the four NN bonds, one SALC each
    return SCEPredictor(b, 0.0, fill(J, 4))
end

# The effective ring coupling (E = J_eff Σ_bonds e_i·e_j), measured from tiled
# energies (the isotropic l = 1 SALC carries a 2√3 normalization — never assume
# the input coefficient): flipping spin 1 flips bonds (1,2) and (4,1) exactly.
function _ring_J(H::MC.TiledHamiltonian)
    up = SVector(0.0, 0.0, 1.0)
    aligned = MC.SpinConfig([up for _ = 1:H.n_sites])
    flipped = copy(aligned)
    flipped[1] = -up
    return (total_energy(H, aligned) - total_energy(H, flipped)) / 4
end

# A genuine higher-multipole two-atom model (l ≤ 2, anisotropic, random couplings)
# — the energy-conservation workhorse.
function _biquadratic_crystal()
    lat = Lattice(Matrix(3.0 * I(3)))
    return Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
end

function _biquadratic_model(seed)
    b = SCEBasis(_biquadratic_crystal(), BasisSpec(; nbody = 2, cutoff = 1.5,
                                                   lmax = [2], isotropy = false))
    return SCEPredictor(b, 0.0, 0.05 .* randn(MersenneTwister(seed), n_salcs(b)))
end

# A single-site uniaxial-anisotropy model (tetragonal cell, spglib backend ⇒ the
# lone body-1 l = 2, m = 0 SALC): E is φ-independent and even in e_z — the 1-D
# Boltzmann quadrature target of the thermostat gate.
function _uniaxial_crystal()
    lat = Lattice(Matrix(Diagonal([4.0, 4.0, 6.0])))
    return Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
end

function _uniaxial_model(K::Float64)
    b = SCEBasis(_uniaxial_crystal(), BasisSpec(; nbody = 1, cutoff = 1.0,
                                                lmax = [2], isotropy = false);
                 backend = SpglibBackend())
    @assert n_salcs(b) == 1
    return SCEPredictor(b, 0.0, [K])
end

_uniaxial_config(u::Float64, φ::Float64 = 0.0) =
    MC.SpinConfig([SVector(sqrt(1 - u^2) * cos(φ), sqrt(1 - u^2) * sin(φ), u)])

# Boltzmann averages of f(u) for a φ-symmetric single-site energy E(u), uniform
# sphere measure (du), by Simpson quadrature.
function _boltzmann_average(Eu, kt::Float64, f; npts::Int = 2001)
    us = range(-1.0, 1.0; length = npts)
    Ev = [Eu(u) for u in us]
    w = exp.(-(Ev .- minimum(Ev)) ./ kt)
    simpson(g) = begin
        @assert isodd(length(g))    # composite Simpson needs an even interval count
        h = Float64(step(us))
        s = g[1] + g[end]
        for i = 2:2:(length(g) - 1)
            s += 4 * g[i]
        end
        for i = 3:2:(length(g) - 2)
            s += 2 * g[i]
        end
        s * h / 3
    end
    return simpson(w .* [f(us[i], Ev[i]) for i in eachindex(us)]) / simpson(w)
end

_rand_spin(rng) = normalize(SVector{3,Float64}(randn(rng, 3)))

_rand_config(rng, H::MC.TiledHamiltonian) =
    MC.SpinConfig([_rand_spin(rng) for _ = 1:H.n_sites])

# Uniform Larmor angular frequency [rad/fs] of a g-spin in |B| tesla.
_larmor_omega(g, B) = g * SD.MU_B_EV_T * B / SD.HBAR_EV_FS
