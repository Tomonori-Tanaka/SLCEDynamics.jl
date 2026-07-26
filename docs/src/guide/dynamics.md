# Deterministic LLG

```@meta
CurrentModule = SLCEDynamics
```

[`run_llg`](@ref) integrates the Landau–Lifshitz–Gilbert equation for unit spin
directions ``\mathbf e_i`` on the SCE Hamiltonian, with fixed step `dt` [fs]:

```math
\frac{d\mathbf e_i}{dt} = p_i\,\bigl[\mathbf e_i \times \mathbf G_i
    + \alpha_i\, \mathbf e_i \times (\mathbf e_i \times \mathbf G_i)\bigr],
\qquad
p_i = \frac{g_i}{\hbar\, \mathrm{magmom}_i\,(1 + \alpha_i^2)},
```

where ``\mathbf G_i = \partial E/\partial \mathbf e_i`` (model energy units) is the
sum of the exact SCE gradient (`SLCEMonteCarlo.energy_gradient!`) and the Zeeman
gradient ``-\mathrm{magmom}_i\,\mu_B\,\mathbf B_\mathrm{ext}``. The torque convention
matches the ecosystem (``\boldsymbol\tau_i = -\mathbf e_i \times \mathbf G_i``, the
physical / Landau–Lifshitz torque), and the damping term strictly dissipates:
``dE/dt = -\sum_i p_i \alpha_i |\mathbf G_{\perp,i}|^2 \le 0``, exactly zero at
``\alpha = 0``.

## Units

- **Energy**: the model's units — eV for DFT-fitted models. **Time**: fs. The two
  meet in [`HBAR_EV_FS`](@ref) (ħ in eV·fs, an exact SI ratio): a gradient magnitude
  ``|G|`` [eV] on a moment `magmom` [μ_B] precesses at ``g|G|/(\hbar\,\mathrm{magmom})``
  [rad/fs].
- **Field**: tesla, only at the boundary — [`MU_B_EV_T`](@ref) converts `b_ext` into
  the Zeeman energy ``E_Z = -\sum_i \mathrm{magmom}_i\,\mu_B\,(\mathbf e_i \cdot
  \mathbf B_\mathrm{ext})``; the core evolution never touches μ_B (it cancels).
  The pinned sanity anchor: a ``g = 2`` spin in 1 T precesses at 27.9925 GHz.

## The problem definition

[`LLGProblem`](@ref)`(H; magmom, alpha = 0.0, g = 2.0, b_ext = (0, 0, 0))` holds the
Hamiltonian plus the per-site material parameters. `magmom` (μ_B, required), `alpha`
(≥ 0), and `g` (> 0) each accept a scalar, a vector with one entry per training-cell
atom (tiled to the supercell), or a full per-site vector. Two conventions inherited
from the siblings:

- **Inactive sites** (outside the fitted magnetic subsystem) are frozen bitwise —
  skipped by both integrator stages, no noise, no Zeeman, excluded from
  `mean_spins`; their parameter entries are ignored.
- With non-uniform `g`, the total moment of a rotation-invariant model is *not*
  conserved at ``\alpha = 0`` (each sublattice precesses at its own rate) — expected
  physics, not an integrator defect.

## Integrators

- [`DepondtMertens`](@ref) (the default) — rotation-based Heun: each stage advances
  every spin by a rigid Rodrigues rotation about its local rotation vector, so
  ``|\mathbf e_i| = 1`` is preserved to rounding with no renormalization. It is
  *exact* for a constant rotation vector, so uniform Larmor precession in a pure
  external field is reproduced to rounding at any step size.
- [`HeunProjected`](@ref) — classical explicit Heun with per-stage renormalization,
  the standard reference scheme of the stochastic-LLG literature. Kept as an
  independent cross-check: two structurally different integrators agreeing to
  ``O(dt^2)`` is a strong implementation gate.

Both are deterministic order 2 and cost two field evaluations per step (≈ two
Metropolis sweeps of the same Hamiltonian).

## Choosing `dt`

Resolve the fastest precession: the largest ``p_i |\mathbf G_i|`` in the run should
satisfy ``\omega_\mathrm{max}\, dt \ll 1`` (a few 0.01–0.1 rad per step). Strongly
coupled models (larger gradients) or small moments (larger ``p``) need smaller
steps; check convergence by halving `dt` — the deterministic error contracts at
``O(dt^2)``. For thermostatted runs see the additional considerations in
[thermal](thermal.md) and [quantum thermostat](quantum_thermostat.md).

## Measurement and renormalization cadence

Measurements are taken at step 0, every `measure_interval` steps, and always at the
final step, so `times[end] == nsteps·dt` and the last measurement describes exactly
the returned `config`. `renorm_interval` (default 1000) renormalizes active spins to
unit length every that many steps: Depondt–Mertens preserves norms to rounding, so
this only caps the ``\sim \varepsilon\sqrt{n_\mathrm{steps}}`` rounding walk of very
long runs (`0` disables; `HeunProjected` renormalizes every step by construction).

## Observables and evaluables

Nothing is hard-coded into the stepping loop: `run_llg` records a vector of
`SLCEMonteCarlo.Observable`s — the **same** definitions the MC drivers accept
(`Observable(name, ncomp, f)` with `f(v::SLCEMonteCarlo.MCView)`), including
`SLCEMonteCarlo.standard_observables(H)` and user-defined ones. Per the `Observable`
contract, `energy` is the **SCE** energy (model units, intercept excluded, Zeeman
not included), so a definition measures identical values on the same configuration
in both packages. Each series lands in `LLGResult.series[name]` as an
`ncomp × n_measurements` matrix; `Evaluable`s enter downstream, through
[`equilibrium_stats`](@ref) on thermostatted runs.

## Energy conservation at ``\alpha = 0``

At zero damping the dynamical energy (SCE + Zeeman) is conserved; the fixed-step
integrator leaves an ``O(dt^2)`` drift. On the cubic ferromagnet of
[Getting started](../getting_started.md):

```@example dyn
using SLCEDynamics, SLCEMonteCarlo, SLCE
import Spglib
using LinearAlgebra, Random

lat = Lattice(Matrix(1.0 * I(3)))
cell = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
basis = SLCEBasis(cell, BasisSpec(; nbody = 2, cutoff = 1.1, lmax = [1],
                                 soc = false);
                 backend = SpglibBackend(), images = AllImages())
model = SLCEModel(basis, 0.0, [-0.01])
H = TiledHamiltonian(model; dims = (2, 2, 2))

config0 = SLCEMonteCarlo.from_matrix(randn(Xoshiro(2), 3, n_sites(H)))
cons = run_llg(LLGProblem(H; magmom = 2.2, alpha = 0.0), config0;
               dt = 0.25, nsteps = 800, measure_interval = 10)
maximum(abs, cons.energies .- cons.energies[1])   # the α = 0 conservation gate
```

With damping the same start dissipates monotonically (compare the largest recorded
energy *rise* against the total drop):

```@example dyn
diss = run_llg(LLGProblem(H; magmom = 2.2, alpha = 0.5), config0;
               dt = 0.25, nsteps = 800, measure_interval = 10)
(drop = diss.energies[end] - diss.energies[1],
 max_rise = maximum(diff(diss.energies)))
```

Both runs are RNG-free and bit-identical for any `ntasks` — rerunning either
reproduces the arrays exactly.
