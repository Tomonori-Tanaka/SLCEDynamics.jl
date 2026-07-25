# The GPU path

```@meta
CurrentModule = SLCEDynamics
```

[`run_llg_gpu`](@ref) runs the same physics on a KernelAbstractions device: both
integrators, deterministic and stochastic (including the quantum thermostat), the
identical `Observable` contract and `LLGResult` semantics — so
[`equilibrium_stats`](@ref), [`structure_factor`](@ref), and checkpointing consume
the result unchanged. It is exported (since the A100 go/no-go and the
quantum-device smoke both passed):

```julia
gH  = GPUTiledHamiltonian(backend, prob.H)     # exported by SLCEMonteCarlo; build ONCE
res = run_llg_gpu(prob, config0, gH; dt = 0.05, nsteps = 10^6,
                  kT = 0.02, seed = 1, checkpoint = "llg_gpu.jld2",
                  checkpoint_interval = 50_000)
```

`gH` is caller-built and reused across runs — the table upload is seconds at
production sizes. The upstream device gradient is `SLCEMonteCarlo`'s
`gpu_energy_gradient!`; there is no CUDA dependency anywhere — the caller passes
the backend object (e.g. `CUDA.CUDABackend()`).

## The determinism contract

Same-seed CPU and GPU runs share the identical stateless Philox noise stream
(`(seed, site, step)` is backend-free), so they are the **same stochastic
realization**, differing only through gradient-fold and rotation roundoff.
The bitwise scope differs from the CPU path:

- A GPU trajectory is bitwise reproducible for a fixed
  (`seed`, backend, `workgroupsize`, package + Julia version). The gradient fold
  order depends on `workgroupsize` (pinned default 128), so unlike the CPU path's
  `ntasks` it **is** part of the contract; the result's `compute` field and the
  checkpoint record all three.
- CPU↔GPU trajectories are therefore never bitwise equal — comparisons are
  tolerance gates on non-chaotic fixtures plus statistical equilibrium/spectral
  gates.
- Checkpoints written on the device resume with `resume(path, prob, gH)`; any
  compute switch requires `allow_compute_switch = true`
  (see [checkpointing](checkpointing.md)).

## KA-CPU as a test backend

`KernelAbstractions.CPU()` runs the full device code path on the host — the CI gate
suite compares it bitwise against a composite keyed reference, no GPU required. It
is also the way to try the API locally:

```@example gpu
using SLCEDynamics, SLCEMonteCarlo, SLCE
import Spglib
using LinearAlgebra, Random
import SLCEDynamics as SD    # only for the KernelAbstractions module below

lat = Lattice(Matrix(1.0 * I(3)))
cell = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
basis = SLCEBasis(cell, BasisSpec(; nbody = 2, cutoff = 1.1, lmax = [1],
                                 isotropy = true);
                 backend = SpglibBackend(), images = AllImages())
model = SLCEModel(basis, 0.0, [-0.01])
H = TiledHamiltonian(model; dims = (2, 2, 2))
prob = LLGProblem(H; magmom = 2.2, alpha = 0.5)
config0 = SLCEMonteCarlo.from_matrix(randn(Xoshiro(1), 3, n_sites(H)))

backend = SD.KernelAbstractions.CPU()          # the KA module SD itself uses
gH = GPUTiledHamiltonian(backend, H)

res1 = run_llg_gpu(prob, config0, gH; dt = 0.5, nsteps = 200,
                   kT = 0.01, seed = 9)
res2 = run_llg_gpu(prob, config0, gH; dt = 0.5, nsteps = 200,
                   kT = 0.01, seed = 9)
(compute = res1.compute, repeat_bitwise = res1.energies == res2.energies)
```

Between measurement/checkpoint events the per-step kernel launches queue with no
synchronization; the loop downloads a host snapshot only when a measurement or a
checkpoint write is due, so a sparse `measure_interval` keeps the device busy.

## Measured performance (A100)

From the decision record `docs/specs/gpu-llg.md` (measured 2026-07-19, kugui
A100-SXM4-40GB): at the l044 8³ production bar (34,816 sites, `workgroupsize =
128`, `dt = 0.05` fs, `kT = 0.02` eV), the LLG step took **178.9 ms** on the device
vs **4471 ms** on the same-node CPU with 8 tasks — **25.0×**, clearing the ≥ 5×
go/no-go. The smoke gates were green (repeat identity bitwise; same-seed CPU-vs-GPU
20-step max deviation 7.6e-16 — ULP-level). Light models are expected marginal and
stay on the CPU path — no performance claims beyond these measurements.

## Quantum thermostat on the device

`run_llg_gpu` accepts `thermostat = QuantumThermostat()` with the same contract as
[`run_llg`](@ref) (temperature required, the same ``k_BT\,\Delta t/\hbar`` bounds):
the stationary initialization always runs on the host and is uploaded once; the
in-kernel cascade filters the same white draws as the CPU path, and the GPU
`resume` restores quantum checkpoint files (coefficients + filter state verbatim).

**Measured cost** (kugui A100, `bench/bench_gpu_qtb.jl` on l044, records in
`docs/specs/quantum-thermostat.md` Q4): the quantum-vs-classical step-time overhead
was within timing noise at 4³ (40.8 vs 40.3 ms, 2026-07-19) and **+0.4%** at the 8³
production size (311.3 vs 312.6 ms, 34,816 sites, 2026-07-25) — the SCE gradient
dominates, so the thermostat choice is effectively free.
