# Semi-quantum (colored-noise) thermostat — decision record

Settled 2026-07-19 by a two-proposal design (physics/numerics + architecture)
plus an adversarial review that adjudicated their one seam conflict and reran
the fit experiments. This file is the condensed authority; the working equation
level detail lives in SPEC.md's "Quantum thermostat" section as it lands.

**Milestone status**: Q-M1 (types + host cascade + counter map + wiring gates)
implemented; Q-M2 checkpoint schema v3 implemented; Q-M3 GPU kernel and
Q-M4 pinned
Barker–Bauer fit constants + physics gates pending. Until Q-M4 the shipped
filter is the **identity placeholder** — a `QuantumThermostat()` run is
bitwise the classical one (deliberate: that identity is the Q-M1 wiring gate),
and the thermostat types stay public-unexported.

## Q1 — target and scope

- Two-sided discrete PSD `σ_i²Δt·θ(ħ|ω|/kT)` on the Nyquist band,
  `θ(x) = x/(eˣ − 1)`, `σ_i` unchanged from the classical FDT (no `(1 + α²)`);
  Barker & Bauer, PRB **100**, 140401(R) (2019). NO zero-point term (ZPE
  leakage through magnon–magnon coupling, and T → 0 must hand back the
  deterministic package — the ecosystem's ground-state anchors).
- Damping stays local/Markovian (the QTB approximation): detailed balance is
  broken; mode occupations are accurate ~1% for `α·x₀ ≲ 0.03` (x₀ = ħω₀/kT),
  ~10% for ≲ 0.3 — production guidance α ≲ 0.05. The classical
  α-independence gate does NOT generalize; quantum gates compare against the
  exact linear-response integral evaluated with the run's own α.
- Validity guard: `τ = kT·Δt/ħ ≤ 0.1` (hard, ArgumentError suggests the
  compliant dt; ≲ 0.05 recommended for ≤ 1% spectral warp at occupied modes).
  Non-binding for kT ≤ 30 meV at dt = 1 fs.

## Q2 — generator: coherent biquad cascade on the EXISTING white draws

- A parallel bank of independent-input sections was rejected with numbers
  (incoherent Σ|H_j|² cannot cancel the Lorentzian 1/x² tail — 5.7% floor;
  matched-Z sampling aliases ~1e-3·σ²Δt). The winner: ONE dimensionless
  rational fit of θ in x = ħω/kT (AAA order ≈ 10, measured 0.9% max rel;
  Froissart-doublet cleanup + positivity refit required — orders 11–13 blow
  up), spectrally factorized to ~5 real biquads, mapped per run by the
  bilinear transform (exact closed-form discrete PSD `θ_fit(x_w)`,
  `x_w = (2/τ)·tan(ωΔt/2)` — no aliasing by construction).
- The DF2T cascade filters the SAME slots-0/1 white triple the classical
  thermostat draws — zero extra per-step Philox blocks, and a same-seed
  classical/quantum pair shares one white realization. Constants ship as
  `const` + `FILTER_ID`; no runtime fitting ever.
- Update once per step in `_fill_noise!`'s slot; both Heun stages read the
  same `gth` (Stratonovich, unchanged).

## Q3 — state, init, counters

- Per-run state: 2 doubles × NS sections × 3 components per site
  (`_FilterState.x`, lane-major `(2·3·NS) × n`, lane =
  `(c − 1)·2NS + 2(j − 1) + r`). Inactive-site columns never drawn, never
  touched — exact 0.0 (the D12 discipline extended to state).
- Stationary init: per component, the 2NS-state discrete Lyapunov equation
  solved by direct vec-solve, clamped symmetric eigendecomposition square root
  (never plain Cholesky — P = 0 exactly for the identity placeholder,
  ULP-negative eigenvalues at small τ), sign-canonicalized so `L` is a
  deterministic function of P; `x₀ = L·ζ` with ζ Philox-keyed at step 0.
- Counter map (word 4 = `_DOMAIN_SD | step_hi16` unchanged; classical
  counters byte-identical to before the feature — pinned by literal Philox
  words in test_quantum_thermostat.jl):

  | purpose                          | step  | slot            |
  |----------------------------------|-------|-----------------|
  | white ξ x,y (both thermostats)   | k ≥ 1 | 0               |
  | white ξ z (4th normal reserved)  | k ≥ 1 | 1               |
  | init ζ block b (ζ[2b+1], ζ[2b+2])| 0     | 2 + b, b < 3·NS |

## Q4 — persistence and compute (Q-M2/Q-M3 contracts)

- Checkpoint schema v3: `run/thermostat` (always), `run/filter_id`
  (provenance only — no refusal), `run/filter/coeffs` (5 × NS, AUTHORITATIVE
  — resume rebuilds the recurrence from the stored coefficients verbatim,
  never from current package constants, so a future refit cannot brick old
  files), `state/filter` (verbatim, the `_config_verbatim` twin);
  persist-not-replay; v1/v2 back-read as classical. Landed (Q-M2): quantum
  checkpoint / crash-resume / extension are gated bit-identical, including a
  non-zero-state writer/reader round-trip (the identity placeholder keeps
  run-produced states at zero, so the layout is pinned directly).
- GPU (Q-M3): device state matrix `n × (6·NS)` (transpose of host — coalesced
  one-thread-per-site; transposed only at event-gated H2D/D2H), literal-port
  noise kernel, init always host-side. Until Q-M3, `run_llg_gpu` has no
  thermostat kwarg (quantum is CPU-only).

## Q5 — statistics boundary

- `LLGResult.thermostat ∈ {"classical", "quantum"}`. `equilibrium_stats` on a
  quantum run THROWS for nonempty `evaluables` unless
  `allow_evaluables = true` (Boltzmann fluctuation formulas are O(1)-wrong
  under QTB; take response functions from d⟨E⟩/dT across runs). S(q,ω)
  machinery is trajectory-level and stays valid; intensities become
  semi-quantum, the detailed-balance asymmetry is NOT reproduced, and the
  deferred classical→quantum intensity factor must never be stacked on
  quantum data. MC cross-validation is classical-only (a deliberate
  ≥ 5σ-mismatch tripwire is part of the Q-M4 gate set).

## Q6 — gate plan (Q-M4)

F1 PSD certificate (closed form, 1e-12 vs the fit + the shipped θ error
bounds); F2 impulse/sinusoid ≡ transfer function; F3 Lyapunov pins; F4 stream
autocovariance 4σ; F5 determinism/bitwise set (already partly live in Q-M1).
G1 Larmor occupation vs the exact α-broadened linear-response integral (3σ,
α ∈ {0.1, 0.5}); G2 Einstein specific heat via two-T finite difference +
`c < 0.6 kB` at x₀ = 3; G3 classical recovery at x₀ ≈ 0.02; G4 dimer
two-mode occupations; G5 MC-mismatch tripwire (≥ 5σ).
