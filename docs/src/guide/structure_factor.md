# The dynamical structure factor S(q,ω)

```@meta
CurrentModule = SLCEDynamics
```

[`structure_factor`](@ref) estimates the classical dynamical spin structure factor
from a recorded trajectory. The estimator (all signs and normalizations pinned by
exact analytic gates — Larmor, dimer, ring dispersion, Parseval sum rules):

```math
\begin{aligned}
s^\alpha(q,t)          &= \tfrac{1}{\sqrt N} \sum_{s\,\mathrm{active}}
                          e^{-2\pi i\, f\cdot x_s}\,(e_s^\alpha(t) - \bar e_s^\alpha),\\
S^{\alpha\beta}(q,\omega_k) &= \tfrac{\Delta t_s}{M W_2}\,
                          X^\alpha(q,\omega_k)^*\, X^\beta(q,\omega_k),
\end{aligned}
```

with ``f`` the fractional q (training-cell r.l.u.), ``x_s`` the fractional site
position, ``\bar e_s`` the per-site time mean over the analysis window, and ``X``
the windowed DFT of ``s``. A spin precessing positively about ``+\hat z`` lands at
``+\omega``; a magnon running toward ``+q`` lands at ``+q``.

## Recording a trajectory

The trajectory is an ordinary observable — [`trajectory_observable`](@ref)`(H)`
records the full configuration (`ncomp = 3·n_sites`), so it inherits the
measurement cadence, checkpoint persistence, resume/extension, and
bit-reproducibility with zero new machinery. Memory guidance: the in-RAM trajectory
is `24·n_sites·n_meas` bytes — keep it ≲ 10 GB (e.g. 8³ cells × 68 atoms × 4096
frames ≈ 3.4 GB). [`trajectory`](@ref) unpacks the series as a zero-copy
`3 × n_sites × n_meas` array (dropping an off-grid final measurement — the
estimator needs a strictly uniform time grid).

## q points and paths

q is fractional, in the training-cell reciprocal lattice; only
supercell-commensurate points (``f_i N_i \in \mathbb Z``) are representable.
`structure_factor` **throws** on incommensurate points, naming the nearest
representable one; [`q_path`](@ref)`(crystal, vertices; npoints, dims = H.dims)`
snaps loudly — it returns both the snapped `qs` and the pre-snap `qs_requested`, so
the snapping is always visible.

## A worked example: magnons on a spin chain

An 8-cell tiling of a ferromagnetic chain along ``x`` (the cell is elongated in
``y``/``z``, so only the ``\pm x`` neighbors are within the cutoff). Thermal
fluctuations populate every magnon mode; light damping keeps the modes sharp:

```@example sqw
using SLCEDynamics, SLCEMonteCarlo, SLCE
import Spglib
using LinearAlgebra, Random

lat = Lattice(Matrix(Diagonal([1.0, 4.0, 4.0])))    # a chain: only ±x in range
cell = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
basis = SLCEBasis(cell, BasisSpec(; nbody = 2, cutoff = 1.1, lmax = [1],
                                 isotropy = true);
                 backend = SpglibBackend(), images = AllImages())
model = SLCEModel(basis, 0.0, [-0.01])
H = TiledHamiltonian(model; dims = (8, 1, 1))       # 8 sites along the chain
prob = LLGProblem(H; magmom = 2.2, alpha = 0.05)
config0 = SLCEMonteCarlo.from_matrix(repeat([0.0, 0.0, 1.0], 1, n_sites(H)))

res = run_llg(prob, config0; dt = 1.0, nsteps = 4096, kT = 0.002, seed = 5,
              measure_interval = 4,
              observables = [trajectory_observable(H)])

path = q_path(cell, [[0.0, 0, 0], [0.5, 0, 0]]; npoints = 5, dims = H.dims)
sqw = structure_factor(res, H, cell, path.qs; discard = 256)
```

The transverse channel ``S^{+-}`` resolves the magnon circulation sense — for a
ferromagnet ordered along ``+\hat z``, all one-magnon weight sits at **positive**
ω, rising from the Goldstone mode at Γ:

```@example sqw
spm = sqw_plusminus(sqw)                       # nq × nω, real ≥ 0
pos = findall(>(0), sqw.omegas)                # the positive-ω half
peaks = [homega_mev(sqw)[pos[argmax(spm[iq, pos])]] for iq = 2:length(path.qs)]
(q = [Tuple(q) for q in path.qs[2:end]], peak_hw_mev = round.(peaks; digits = 2))
```

## Estimator controls

- **Welch averaging**: `nsegments` segments of derived power-of-two length `nfft`
  (`seglength` overrides; trailing samples are truncated, never zero-padded), with
  fractional `overlap` and a `:hann` (default) or `:none` window.
- **`discard`** drops leading measurements as thermalization.
- **Mean/elastic separation**: the per-site time mean over the analysis window (one
  global mean, never per segment) is always subtracted; the elastic tensor is
  reported separately in `S_el`, so Bragg weight cannot leak through the window
  into the inelastic spectrum.
- The result is bit-identical for any `ntasks` and across repeated calls.

!!! note "Windowing caveat"
    Hann windowing plus mean subtraction still leaves a residual low-ω leak on
    thermal data — inspect the near-elastic bins accordingly.

## The result and its reductions

[`SQWResult`](@ref) stores the full 3×3 Hermitian tensor `S[α, β, iq, iω]` on a
two-sided fftshifted frequency axis (`omegas` [rad/fs]; for even `nfft` the Nyquist
bin exists only on the negative side), plus the elastic tensor and the estimation
metadata. Reductions are cheap post-contractions:

- [`sqw_diag`](@ref) — the three diagonal components;
- [`sqw_trace`](@ref) — ``\sum_\alpha S^{\alpha\alpha}``, the rotation-invariant
  total intensity (the sum-rule object);
- [`sqw_perp`](@ref) — the unpolarized-neutron weight (`NaN` at Γ, where the
  direction is undefined — no silent convention);
- [`sqw_plusminus`](@ref) — the ω-sign-resolving transverse channel;
- [`sqw_elastic`](@ref) — the elastic tensor.

Axis helpers: [`homega_ev`](@ref) / [`homega_mev`](@ref) (``\hbar\omega``) and
[`freq_thz`](@ref).

## Ensembles and files

`structure_factor(::Vector{LLGResult}, …)` averages a seed ensemble (spectra, never
amplitudes) and reports per-element realization standard errors in `err` for ≥ 3
members; members must share `dt`, `nsteps`, and `measure_interval`.
`structure_factor(path::AbstractString, …)` reads a [`run_llg`](@ref) checkpoint
file directly — a completed checkpoint *is* the persisted trajectory format.

## Verification: the channel sum rule

`channel_sumrule` (public, unexported) checks the global Parseval identity of the
estimator at the sublattice-channel level against the direct-space
``\sum_{s\,\mathrm{active}} (1 - |\bar e_s|^2)`` — a plumbing gate valid for any
trajectory:

```@example sqw
tr = trajectory(res)
SLCEDynamics.channel_sumrule(tr.traj, tr.times, H, cell)
```
