# GPU LLG — decision record

> **Naming note (2026-07-28).** This is a dated decision record and is kept as
> written; the names below are the ones the decision was taken under. Renamed
> since, in the family-wide naming batch: `run_llg_gpu` → `gpu_run_llg`. The current spelling is what
> the code, `SPEC.md` and the API reference use.


Settled by a two-designer + adversarial-review pass (2026-07-19); the upstream
half (the device all-site SCE gradient) is SLCEMonteCarlo's G7 record
(`docs/specs/gpu-prototype.md` there). This file records the consumer-side
decisions; implement/change against it.

## L1 — API and boundary

- **Separate entry point `run_llg_gpu(prob, config0, gH; …)`** (public,
  unexported until the A100 go/no-go, mirroring the sibling's
  `gpu_run_sweeps!`; **promoted to export 2026-07-19** after the GO and the
  quantum-device smoke both landed — `GPULLGState` stays public-unexported): a `backend` kwarg on `run_llg` would silently weaken its
  documented "pure function of `(prob, config0, dt, nsteps, integrator, seed)`,
  bit-identical for any `ntasks`" contract, which the GPU path cannot honor —
  its trajectory additionally depends on (backend, `workgroupsize`).
- `gH = SLCEMonteCarlo.GPUTiledHamiltonian(backend, prob.H)` is caller-built and
  reused (table upload is seconds at production sizes); validated against
  `prob.H` (identity or `model_fingerprint`).
- The upstream boundary: `SLCEMonteCarlo.gpu_energy_gradient!` (+
  `GPUGradientScratch`) delivers ONLY the tangent-projected SCE gradient; SD's
  stage kernels add the unprojected `gzee` and the thermal `gth` exactly as the
  host `_omega` does. No CUDA dependency anywhere (KernelAbstractions hard dep;
  the caller passes the backend — the sibling's convention; kugui's
  machine-global CUDA pin applies unchanged).

## L2 — device step and state

- Per Depondt step, 7 queued launches (noise; rows+gradient ×2 per stage;
  stage 1; stage 2), **no fusion** across the package boundary and **no
  synchronization** — KA queue order serializes; the loop syncs only before a
  host download (measurement/checkpoint events, `_ck_due` mirroring
  `_ck_llg!`'s cadence exactly). Every gradient call refreshes the tesseral
  rows (`refresh_zrows = true`): both stages move every spin, so no row reuse
  is ever valid — no stale-rows hazard by construction.
- `GPULLGState` is SD-owned (`_LLGScratch`'s device sibling + the per-site
  problem arrays + the upstream gradient scratch + a host snapshot buffer);
  `GPUChainState` is deliberately not reused (MC bookkeeping).
- The stage kernels are literal ports of `_omega`/`_rotate`/the stage
  structure (one thread per site, same expression order); the renorm kernel is
  IEEE-exact and bitwise-equal to the host `_renormalize_active!`. Inactive
  sites are gated by the SD-uploaded `dactive` mask in every SD kernel (the
  gradient needs no mask — empty adjacency folds to an exact zero upstream).
- **The noise kernel branches on `dactive` and writes an exact
  `zero(SVector)`** for inactive sites: computing `σ·ξ` with `σ = 0` would
  produce −0.0 components under negative normals and break the bitwise gate
  against the host `_fill_noise!` buffer (D12).

## L3 — determinism ledger and validation tiers

Same-seed CPU and GPU runs share the identical Philox noise stream (`(seed,
site, step)` is stateless and backend-free), so they are the SAME stochastic
realization, differing only through gradient-fold and rotation roundoff (and
transcendental ULPs on CUDA). What is bitwise, per backend pair:

| Quantity | CPU path vs KA-CPU device | CPU path vs CUDA |
|---|---|---|
| philox blocks / uniforms | bitwise | bitwise |
| Box–Muller normals | bitwise (same libm) | ~1 ulp |
| renorm / projection / sqrt | bitwise | bitwise |
| `_rotate` (`sincos`) | bitwise | ULP |
| SCE gradient | NOT bitwise (fold order) | NOT bitwise |
| full trajectory | NOT bitwise | NOT bitwise |

Therefore: the bitwise tier compares the device path against the **composite
keyed reference** (upstream `_gradient_lane_ref!` + the literal stage
expressions — `test_gpu_llg.jl`), never against `run_llg`. The GPU contract:
bitwise reproducible for fixed (`seed`, backend, `workgroupsize`, package +
Julia version); ws is part of the tuple because the gradient fold depends on
it (pinned default 128; note a site with fewer adjacency entries than ws folds
identically for any ws — ws-sensitivity gates need entry-rich models).
CPU↔GPU comparisons are tolerance gates on non-chaotic fixtures (linear
roundoff envelope) plus statistical equilibrium/spectral gates; same-seed
common-random-number pairing is valid only within a decorrelation time.

## L4 — checkpoint schema v2 and cross-compute resume

- Schema v2 adds `run/compute` (`"cpu"`/`"gpu"`), `run/backend` (a stable
  lowercase tag via `_backend_tag`, validation-only — the backend object always
  comes from the caller's `gH`), `run/workgroupsize` (0 on CPU). v1 files are
  back-read as (`"cpu"`, `""`, 0).
- `resume(path, prob)` refuses GPU files, naming the device method;
  `resume(path, prob, gH; …)` continues bit-identically for the same
  (compute, backend tag, ws), including `nsteps` extension. **Any switch
  errors unless `allow_compute_switch = true`** — the continuation is then the
  same physical realization (verbatim config + the same noise stream) but not
  bit-identical to any single-backend run; the opt-in is deliberate (a silent
  backend switch once burned a walltime — sibling's G6 record).

## L5 — measurements and scope

- Measurements/checkpoints run on a host snapshot (one D2H per event; the
  identical `_measure!`/`Observable` contract, so `LLGResult`,
  `equilibrium_stats`, `structure_factor`, and the trajectory observable work
  unchanged — S(q,ω) gates on `run_llg_gpu(CPU())` trajectories pass verbatim).
- v1 in: both integrators, deterministic + sLLG, device renorm, checkpoint v2 +
  resume, the KA-CPU gate suite, the A100 bench script. Deferred: device-side
  observable reductions, async-overlap measurement, GPU S(q,ω), multi-GPU,
  Float32.
- **No performance claims until measured** (the settled blocker): the go/no-go
  is the A100 bench's ≥ 5× at the l044 8³ LLG step (l02 is expected marginal
  and is NOT a bar — light models stay on the CPU path). `bench/bench_gpu_llg.jl`
  runs the smoke gates (repeat identity, b-tier tolerances) before timing.
- **Measured 2026-07-19 (kugui A100-SXM4-40GB, job 858227) — GO.** l044 8³
  (34,816 sites, ws = 128, dt = 0.05 fs, kT = 0.02 eV): smoke green (repeat
  identity bitwise; same-seed CPU-vs-GPU 20-step max deviation 7.6e-16 —
  ULP-level, matching the sibling's CUDA Box–Muller scope); LLG step
  **178.9 ms** on the device vs **4471 ms** on the same-node CPU (8 tasks) —
  **25.0×** at the l044 8³ bar (≥ 5× ⇒ GO). Consistent with the sibling's
  l044 sweep cost (one step = two gradient evals ≈ two sweeps). The upstream
  GR9 gradient-kernel claim held bitwise on CUDA the same job.
