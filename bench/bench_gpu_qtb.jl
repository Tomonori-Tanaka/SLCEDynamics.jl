# A100 smoke for the quantum-thermostat device path (decision record
# docs/specs/quantum-thermostat.md Q4; run on kugui's accelerator queue with
# SCE_REQUIRE_CUDA=1).
#
#   julia --project=bench -t 8 bench/bench_gpu_qtb.jl <model.toml> [dims]
#
# `bench/Project.toml` is this package's own GPU bench environment: it carries
# CUDA, which the shared `@slce` env deliberately does not. Path-dev the three
# SLCE packages into it once per machine (the Manifest is gitignored).
#
# Sections: (1) classical run with the zero-sized filter allocations (the
# refactor must not disturb the classical device path); (2) quantum end-to-end
# on the device (repeat identity, quantum != classical, same-seed CPU-vs-GPU
# short-horizon tolerance); (3) device checkpoint/resume/extension bitwise
# identity for the quantum filter round-trip; (4) classical-vs-quantum step
# timing (cascade overhead readout).

using SLCE, SLCEMonteCarlo, SLCEDynamics
using KernelAbstractions: KernelAbstractions, CPU
using StaticArrays, LinearAlgebra, Printf, Random
const MC = SLCEMonteCarlo
const SD = SLCEDynamics

const REQUIRE_CUDA = get(ENV, "SCE_REQUIRE_CUDA", "0") == "1"
HAVE_CUDA = false
backend = CPU()
try
    @eval using CUDA
    if CUDA.functional()
        global HAVE_CUDA = true
        global backend = CUDABackend()
        CUDA.versioninfo()
    end
catch err
    @warn "CUDA unavailable" err
end
REQUIRE_CUDA && !HAVE_CUDA && error("SCE_REQUIRE_CUDA=1 but CUDA is not functional")

modelpath = length(ARGS) >= 1 ? ARGS[1] : error("usage: bench_gpu_qtb.jl model.toml [n]")
n_dim = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
model = SLCE.load(SLCEModel, modelpath)
H = TiledHamiltonian(model; dims = (n_dim, n_dim, n_dim))
println("model: ", modelpath, "  dims = ", (n_dim, n_dim, n_dim),
        "  n_sites = ", H.n_sites)
flush(stdout)

rng = Xoshiro(7)
c0 = MC.SpinConfig([normalize(SVector{3,Float64}(randn(rng, 3)))
                    for _ = 1:H.n_sites])
prob = LLGProblem(H; magmom = 2.0, alpha = 0.5)
gH = MC.GPUTiledHamiltonian(backend, H)
# kT*dt/hbar ~ 1.5e-3 — inside the quantum thermostat's validity window.
kw = (; dt = 0.05, kT = 0.02, seed = 11, measure_interval = 10)
qt = (; thermostat = QuantumThermostat())

# --- 1. classical with zero-sized filter allocations ---------------------------------
rc1 = SD.run_llg_gpu(prob, c0, gH; kw..., nsteps = 20)
rc2 = SD.run_llg_gpu(prob, c0, gH; kw..., nsteps = 20)
rc1.config == rc2.config || error("classical repeat identity FAILED")
rc_cpu = run_llg(prob, c0; kw..., nsteps = 20)
devc = maximum(norm.(rc_cpu.config .- rc1.config))
println("classical smoke: repeat identity OK; CPU-vs-GPU 20-step max dev = ", devc)
devc <= 1e-6 || error("classical same-seed deviation implausibly large")
flush(stdout)

# --- 2. quantum end-to-end -----------------------------------------------------------
rq1 = SD.run_llg_gpu(prob, c0, gH; kw..., qt..., nsteps = 20)
rq2 = SD.run_llg_gpu(prob, c0, gH; kw..., qt..., nsteps = 20)
rq1.config == rq2.config || error("quantum repeat identity FAILED")
rq1.config != rc1.config || error("quantum config identical to classical — filter inert?")
rq_cpu = run_llg(prob, c0; kw..., qt..., nsteps = 20)
devq = maximum(norm.(rq_cpu.config .- rq1.config))
println("quantum smoke: repeat identity OK; CPU-vs-GPU 20-step max dev = ", devq,
        HAVE_CUDA ? " (ULP-level expected)" : " (KA-CPU)")
devq <= 1e-6 || error("quantum same-seed deviation implausibly large")
flush(stdout)

# --- 3. device checkpoint/resume bitwise ---------------------------------------------
tmp = mktempdir()
ckpath = joinpath(tmp, "qtb.jld2")
ra = SD.run_llg_gpu(prob, c0, gH; kw..., qt..., nsteps = 40)
SD.run_llg_gpu(prob, c0, gH; kw..., qt..., nsteps = 20,
               checkpoint = ckpath, checkpoint_interval = 20)
rb = MC.resume(ckpath, prob, gH; nsteps = 40)
ra.config == rb.config || error("quantum device resume NOT bitwise")
println("quantum device checkpoint/resume/extension: bitwise OK")
flush(stdout)

# --- 4. timing -----------------------------------------------------------------------
nsteps_t = 50
SD.run_llg_gpu(prob, c0, gH; kw..., nsteps = 5)          # warmup/compile
SD.run_llg_gpu(prob, c0, gH; kw..., qt..., nsteps = 5)
t_c = @elapsed SD.run_llg_gpu(prob, c0, gH; kw..., nsteps = nsteps_t)
t_q = @elapsed SD.run_llg_gpu(prob, c0, gH; kw..., qt..., nsteps = nsteps_t)
@printf("GPU step: classical %.1f ms  quantum %.1f ms  overhead %+.1f%%\n",
        1e3 * t_c / nsteps_t, 1e3 * t_q / nsteps_t, 100 * (t_q / t_c - 1))

if HAVE_CUDA
    println("=== quantum-thermostat A100 smoke: ALL SECTIONS PASSED ===")
else
    println("(KA-CPU smoke only — rerun with a CUDA device for the Q4 record)")
end
flush(stdout)
