# API reference

```@meta
CurrentModule = SLCEDynamics
```

## Module

```@docs
SLCEDynamics
```

## Units

```@docs
HBAR_EV_FS
MU_B_EV_T
```

## Problem definition

```@docs
LLGProblem
SLCEMonteCarlo.total_energy(::LLGProblem, ::SpinConfig)
```

## Integrators

```@docs
DepondtMertens
HeunProjected
```

`AbstractIntegrator` (public, unexported) is the integrator union
`Union{DepondtMertens, HeunProjected}` — the type of `run_llg`'s `integrator`
keyword and the dispatch seam for future schemes.

## Running

```@docs
run_llg
LLGResult
```

## Thermostats

```@docs
ClassicalThermostat
QuantumThermostat
```

`AbstractThermostat` (public, unexported) is the thermostat union
`Union{ClassicalThermostat, QuantumThermostat}` — the type of the drivers'
`thermostat` keyword.

## Equilibrium statistics

```@docs
equilibrium_stats
```

## S(q,ω)

```@docs
trajectory_observable
trajectory
q_path
structure_factor
SQWResult
sqw_diag
sqw_trace
sqw_perp
sqw_plusminus
sqw_elastic
homega_ev
homega_mev
freq_thz
```

## Checkpoint / resume

```@docs
resume(::AbstractString, ::LLGProblem)
```

## GPU

```@docs
run_llg_gpu
resume(::AbstractString, ::LLGProblem, ::Any)
```

## Re-exported from SLCEMonteCarlo

`Observable`, `Evaluable`, and `resume` are the **same bindings** as
`SLCEMonteCarlo`'s (re-exported here): observable/evaluable definitions written for
the MC drivers plug into this package unchanged, and `resume` gains the
`LLGProblem` methods documented on this page. See the SLCEMonteCarlo API reference
for the `Observable`/`Evaluable` contracts.

## Public, unexported

Internal-but-stable machinery is public without being exported — call it
qualified (`SLCEDynamics.GPULLGState`, …).

```@docs
GPULLGState
ColoredNoiseFilter
channel_sumrule
```
