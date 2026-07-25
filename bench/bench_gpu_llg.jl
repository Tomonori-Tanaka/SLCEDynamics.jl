# A100 smoke + bench for the GPU LLG path (decision record docs/specs/gpu-llg.md
# L5; run on kugui's accelerator queue with SCE_REQUIRE_CUDA=1).
#
#   julia --project=@slce -t 8 bench/bench_gpu_llg.jl <model.toml> [dims]
#
# Sections: (1) correctness smoke on the device (repeat identity + same-seed
# CPU-vs-GPU short-horizon tolerance); (2) T_step timing CPU vs GPU; (3) the
# go/no-go readout (≥ 5× at the production model — l044 8³ is the bar).

using SLCE, SLCEMonteCarlo, SLCEDynamics
using KernelAbstractions
using StaticArrays, LinearAlgebra, Printf, Random
const MC = SLCEMonteCarlo
const SD = SLCEDynamics

const REQUIRE_CUDA = get(ENV, "SCE_REQUIRE_CUDA", "0") == "1"
HAVE_CUDA = false
BACKEND = CPU()
try
    @eval using CUDA
    if CUDA.functional()
        global HAVE_CUDA = true
        global BACKEND = CUDABackend()
        CUDA.versioninfo()
    end
catch err
    @warn "CUDA unavailable" err
end
REQUIRE_CUDA && !HAVE_CUDA && error("SCE_REQUIRE_CUDA=1 but CUDA is not functional")

modelpath = length(ARGS) >= 1 ? ARGS[1] : error("usage: bench_gpu_llg.jl model.toml [n]")
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
gH = MC.GPUTiledHamiltonian(BACKEND, H)
kw = (; dt = 0.05, kT = 0.02, seed = 11, measure_interval = 1000)

# --- 1. smoke ------------------------------------------------------------------------
r1 = SD.run_llg_gpu(prob, c0, gH; kw..., nsteps = 20)
r2 = SD.run_llg_gpu(prob, c0, gH; kw..., nsteps = 20)
r1.config == r2.config || error("repeat identity FAILED — do not bench")
rc = run_llg(prob, c0; kw..., nsteps = 20)
dev = maximum(norm.(rc.config .- r1.config))
println("smoke: repeat identity OK; same-seed CPU-vs-GPU 20-step max dev = ",
        dev, HAVE_CUDA ? " (ULP-level expected)" : " (KA-CPU)")
dev <= 1e-6 || error("same-seed short-horizon deviation implausibly large")
flush(stdout)

# --- 2. timing -----------------------------------------------------------------------
nsteps_gpu = 50
SD.run_llg_gpu(prob, c0, gH; kw..., nsteps = 5)          # warmup/compile
t_gpu = @elapsed SD.run_llg_gpu(prob, c0, gH; kw..., nsteps = nsteps_gpu)
gpu_ms = 1e3 * t_gpu / nsteps_gpu
nsteps_cpu = max(2, round(Int, 5_000 / H.n_sites))
run_llg(prob, c0; kw..., nsteps = 1, ntasks = Threads.nthreads())
t_cpu = @elapsed run_llg(prob, c0; kw..., nsteps = nsteps_cpu,
                         ntasks = Threads.nthreads())
cpu_ms = 1e3 * t_cpu / nsteps_cpu
@printf("LLG step: cpu-%dT %.1f ms  gpu %.1f ms  ratio %.1fx\n",
        Threads.nthreads(), cpu_ms, gpu_ms, cpu_ms / gpu_ms)

# --- 3. go/no-go ---------------------------------------------------------------------
if HAVE_CUDA
    verdict = cpu_ms / gpu_ms >= 5 ? "GO (≥ 5x)" : "NO-GO (< 5x — keep CPU)"
    @printf("=== GPU LLG go/no-go @ %d³: %.1fx → %s ===\n", n_dim,
            cpu_ms / gpu_ms, verdict)
else
    println("(KA-CPU smoke only — no go/no-go readout without a CUDA device)")
end
flush(stdout)
