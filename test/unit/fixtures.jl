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

# A genuine higher-multipole two-atom model (l ≤ 2, anisotropic, random couplings)
# — the energy-conservation workhorse.
function _biquadratic_model(seed)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    b = SCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2],
                               isotropy = false))
    return SCEPredictor(b, 0.0, 0.05 .* randn(MersenneTwister(seed), n_salcs(b)))
end

_rand_spin(rng) = normalize(SVector{3,Float64}(randn(rng, 3)))

_rand_config(rng, H::MC.TiledHamiltonian) =
    MC.SpinConfig([_rand_spin(rng) for _ = 1:H.n_sites])

# Uniform Larmor angular frequency [rad/fs] of a g-spin in |B| tesla.
_larmor_omega(g, B) = g * SD.MU_B_EV_T * B / SD.HBAR_EV_FS
