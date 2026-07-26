# Bloch-law magnetization example (decision record docs/specs/quantum-thermostat.md,
# gate tier G6): M(T) of a small cubic Heisenberg ferromagnet under the classical
# and the quantum (colored-noise) thermostat, against linear spin-wave estimates.
#
# Physics: classically every magnon mode carries kT, so the low-T magnetization
# deficit 1 − M = (1/N)·Σ_q kT/b_q is LINEAR in T. Quantum-mechanically the
# high-ω modes freeze out (occupation θ(x) = x/(eˣ−1), no zero point), which is
# what produces the Bloch T^{3/2} law in the continuum limit; on this discrete
# 4³ grid the visible signature is the strong low-T flattening of 1 − M.
#
#   julia --project -t 4 examples/bloch_mt.jl [dims] [nsteps]
#
# Runtime ≈ a few minutes at the defaults (dims = 4, nsteps = 400_000; six
# thermostatted runs). All runs are seeded — the printed table is reproducible.
#
# Estimator notes (the semi-quantum statistics boundary, record Q1/Q5):
# the "LSWT quantum" column uses the IDEAL occupation kT·θ(ħω_q/kT) per mode —
# it neglects the α-broadening of the mode Lorentzians and the shipped-filter
# fit error (both are part of the measured value; see test/unit/qt_predictions.jl
# for the exact machinery), so a few-percent gap to the measurement is expected.

using SLCEDynamics, SLCEMonteCarlo, SLCE
using Spglib: Spglib          # activates SLCE's SpglibBackend extension
using LinearAlgebra, StaticArrays

const MC = SLCEMonteCarlo

dims = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
nsteps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 400_000

# --- model: 1-atom cubic cell, NN Heisenberg (the documentation model) --------
lat = Lattice(Matrix(1.0 * I(3)))
cell = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
basis = SLCEBasis(cell, BasisSpec(; nbody = 2, cutoff = 1.1, lmax = [1],
                                 soc = false);
                 backend = SpglibBackend(), images = AllImages())
model = SLCEModel(basis, 0.0, fill(-0.01, n_salcs(basis)))
H = TiledHamiltonian(model; dims = (dims, dims, dims))
N = n_sites(H)

# Effective NN coupling J_eff (E = J_eff Σ_bonds e_i·e_j), measured from tiled
# energies — never assume the SALC normalization. Flipping one spin flips its
# 6 bonds: ΔE = 12·|J_eff|.
up = SVector(0.0, 0.0, 1.0)
aligned = MC.SpinConfig([up for _ = 1:N])
flipped = copy(aligned)
flipped[1] = -up
Jeff = (total_energy(H, aligned) - total_energy(H, flipped)) / 12
@assert Jeff < 0

# --- parameters ---------------------------------------------------------------
g = 2.0
magmom = 20.0                     # large μ keeps the dynamics linear (b ≫ kT)
alpha = 0.05                      # the recommended quantum-thermostat regime
dt = 1.0                          # fs; τ = kT·dt/ħ stays inside [1e-4, 0.1]
b = 0.02                          # Zeeman gap [eV] — pins ẑ, gaps the q = 0 mode
Bz = b / (magmom * MU_B_EV_T)
kts = [0.002, 0.005, 0.01]        # ħω̃ spans 0.002 (gap) … ~0.30 eV (zone corner)

# --- linear spin-wave estimates ----------------------------------------------
# Mode stiffnesses on the discrete grid: b_q = b + 2|J_eff|·Σ_a (1 − cos q_a).
theta(x) = x == 0.0 ? 1.0 : x / expm1(x)
bqs = [b + 2 * abs(Jeff) * ((1 - cospi(2 * m1 / dims)) +
                            (1 - cospi(2 * m2 / dims)) +
                            (1 - cospi(2 * m3 / dims)))
       for m1 = 0:(dims - 1), m2 = 0:(dims - 1), m3 = 0:(dims - 1)]
# ⟨1 − e_z⟩ per site = (1/N)·Σ_q ⟨E_q⟩/b_q; ħω̃_q = (g/μ)·b_q.
deficit_cl(kt) = sum(kt / bq for bq in bqs) / N
deficit_qt(kt) = sum(kt * theta(g * bq / (magmom * kt)) / bq for bq in bqs) / N

# --- runs ---------------------------------------------------------------------
prob = LLGProblem(H; magmom, alpha, b_ext = (0.0, 0.0, Bz))
mobs = [Observable(:mz, 1, v -> sum(s[3] for s in v.config) / length(v.config))]

function measure_m(thermostat, kt, seed)
    res = run_llg(prob, MC.SpinConfig([up for _ = 1:N]); dt, nsteps, kT = kt,
                  seed, measure_interval = 40, observables = mobs, thermostat)
    st = equilibrium_stats(res; evaluables = Evaluable[])[:mz]
    return st.mean[1], st.err[1]
end

println("cubic ferro $(dims)³ ($(N) sites), J_eff = $(round(Jeff; sigdigits = 4)) eV,",
        " gap ħω̃₀ = $(round(g * b / magmom; sigdigits = 3)) eV, zone-corner ħω̃ = ",
        "$(round(g * (b + 12 * abs(Jeff)) / magmom; sigdigits = 3)) eV")
println()
println(" kT [eV]  | M classical (LSWT)      | M quantum (LSWT ideal θ)")
println("----------+-------------------------+-------------------------")
for (i, kt) in enumerate(kts)
    mc, ec = measure_m(ClassicalThermostat(), kt, 100 + i)
    mq, eq = measure_m(QuantumThermostat(), kt, 200 + i)
    println(" ", rpad(kt, 8), " | ",
            rpad("$(round(mc; digits = 4))±$(round(ec; sigdigits = 2))", 15),
            rpad("($(round(1 - deficit_cl(kt); digits = 4)))", 9), "| ",
            rpad("$(round(mq; digits = 4))±$(round(eq; sigdigits = 2))", 15),
            "($(round(1 - deficit_qt(kt); digits = 4)))")
end
println()
println("classical deficit 1 − M is linear in kT; the quantum deficit collapses at")
println("low kT as the magnon band freezes out — the discrete-grid analogue of the")
println("Bloch T^{3/2} law. (Parenthesised: linear spin-wave estimates.)")
