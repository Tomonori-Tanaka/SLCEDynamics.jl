# Checkpointing and restart

```@meta
CurrentModule = SLCEDynamics
```

Long runs checkpoint to a JLD2 file and restart **bit-identically** — the resumed
trajectory, measurement arrays, and final configuration equal the uninterrupted
run's exactly (tested with `==`, not `≈`).

```julia
run_llg(prob, config0; dt = 0.5, nsteps = 10^7, kT = 0.01, seed = 1,
        checkpoint = "llg.jld2", checkpoint_interval = 50_000)

# …after a crash / walltime kill:
result = resume("llg.jld2", prob)      # returns the FULL run's LLGResult
```

With a path, a checkpoint is written every `checkpoint_interval` steps (`0` ⇒ only
at completion) and always once at completion. Writes are atomic (temp file + `mv`)
and consume no RNG, so checkpointing never perturbs the run it protects. One writer
per checkpoint path — two concurrent runs must not share one.

## The resume contract

- The caller re-supplies the `LLGProblem` and any custom `observables` — function
  objects are not serialized. The file stores the **model fingerprint**
  (`SLCEMonteCarlo.model_fingerprint`), the problem parameters
  (`magmom`/`alpha`/`g`/`b_ext`, compared on active sites), and the observable
  names/component counts, and errors on any mismatch — a resume against different
  physics never silently continues.
- Everything trajectory-defining (`dt`, `measure_interval`, `renorm_interval`,
  integrator, `kT`, `seed`, thermostat) comes from the file and cannot be
  overridden.
- `nsteps` larger than the stored one **extends** the run — bit-identical to a
  single uninterrupted run of that length. Extension past a completed run whose
  final measurement was off the `measure_interval` grid is refused (its trace would
  not be a prefix of the longer run's).
- Resuming a *completed* run reconstructs its `LLGResult` without stepping —
  `resume` is idempotent, so a blind "resubmit and `resume` if the file exists"
  retry loop is safe.
- By default the resumed run keeps checkpointing to the same path with the stored
  cadence (`checkpoint = nothing` disables, `checkpoint_interval` overrides).

## Why the files are small and the restart bitwise

The thermal noise is a stateless pure function of `(seed, site, step)`, so **no RNG
state exists to store** — unlike the MC checkpoints, an LLG file carries no
generator words at all. The configuration is restored *verbatim* (deliberately not
through `SLCEMonteCarlo.from_matrix`, whose renormalization would perturb a chaotic
trajectory by ULPs), and every per-step effect — noise counter, renormalization
cadence, measurement grid — is a pure function of the absolute step index.

## Schema v3 contents

Plain data only (no Julia struct reconstruction), in named JLD2 groups: the schema
version and kind (`"llg"`), the model fingerprint, the problem arrays, the run
parameters (`dt`, `nsteps`, `measure_interval`, `renorm_interval`, integrator name,
`kT`, `seed`), the compute provenance (`compute`/`backend`/`workgroupsize` — v2),
the thermostat tag plus — for quantum runs — the filter provenance id, the
**authoritative** filter coefficients, and the filter state restored verbatim (the
one carried per-step array; v3), the completed step, the bitwise configuration, and
the measurements recorded so far. v1/v2 files are back-read (as `"cpu"` /
`"classical"`).

## Compute-switch policy

`resume(path, prob)` refuses a checkpoint written by `gpu_run_llg`, naming the
device method; `resume(path, prob, gH)` continues bit-identically for the same
(backend, `workgroupsize`) and refuses **any** compute switch — a CPU file, a
different backend tag, or a different workgroup size — unless
`allow_compute_switch = true`. When allowed, the continuation restores the
configuration verbatim and the same noise stream, so it is the same physical
realization continued exactly from a valid state — but not bit-identical to any
single-backend run; the opt-in is deliberately loud. See [GPU](gpu.md).
