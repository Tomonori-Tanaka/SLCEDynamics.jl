# Semi-quantum (colored-noise) thermostat — decision record

Settled 2026-07-19 by a two-proposal design (physics/numerics + architecture)
plus an adversarial review that adjudicated their one seam conflict and reran
the fit experiments. This file is the condensed authority; the working equation
level detail lives in SPEC.md's "Quantum thermostat" section as it lands.

**Milestone status**: ALL FOUR milestones implemented (Q-M1 types/cascade,
Q-M2 checkpoint schema v3, Q-M3 GPU kernel, Q-M4 pinned constants + physics
gates). `ClassicalThermostat`/`QuantumThermostat` are exported.

**Shipped filter** (`_QT_S_BIQUADS`, `_QT_FILTER_ID = "bb-aaa10-lm-v1"`,
fitted by the committed deterministic `dev/fit_qtb_filter.jl` — rerunning
reproduces the constants exactly): 4 DC-normalized biquads (order 8; one
Froissart doublet removed from the AAA-10 candidate). Measured certificate:
max rel 4.11e-3 on x ∈ [0.01, 6.6] (gate 1.5e-2), max abs 7.17e-5 on
(6.6, 200] (gate 3e-4), H(0) = 1 exact (per-section β0 ≡ α0 — survives the
bilinear map), |H|² > 0 structurally (tail-notch zero pair at
−7.7e-5 ± 13.43i, floor ~7e-14), bilinear identity + H_d(1) = 1 verified in
256-bit arithmetic, z-poles inside the unit disk over τ ∈ [0.001, 0.1].
Float64 z-form conditioning near DC at τ ≤ 0.01 is ~1e-6 — runtime gates
bound Pd(0) at 1e-6, never 1e-12 (the exact claims are the script's).

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
  rational fit of θ in x = ħω/kT (AAA order ≈ 10, ~0.9% pre-polish, refined
  to 0.41% by the LM step;
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
- GPU (Q-M3, landed): device state matrix `n × (6·NS)` (transpose of host —
  coalesced one-thread-per-site; transposed only at the event-gated H2D/D2H,
  a bitwise-lossless permutation staged through `h_xstage`), literal-port
  `_noise_kernel_quantum!`/`_qt_cascade_dev!`, stationary init always
  host-side and uploaded once. `run_llg_gpu` takes the same `thermostat`
  kwarg/validation as `run_llg`; the GPU `resume` restores quantum files
  (coeffs + state verbatim). KA-CPU gates: kernel ≡ host fill bitwise (gth +
  full state, nontrivial 2-section filter), device identity wiring gate,
  GPU quantum checkpoint/resume/extension bitwise.
- **A100 smoke: PASSED** (2026-07-19, kugui i1accs job 858230,
  A100-SXM4-40GB, CUDA 12.6, `bench/bench_gpu_qtb.jl` on l044 at 4³,
  `dt = 0.05` fs, `kT = 0.02` eV, ws = 128). All sections green on the real
  device: classical run with the zero-sized filter allocations (repeat
  identity; CPU-vs-GPU 20-step max dev 6.8e-16), quantum end-to-end (repeat
  identity; dev 5.0e-16, ULP-level; config ≠ classical), device quantum
  checkpoint/resume/extension bitwise. Step time at 4³: classical 40.8 ms,
  quantum 40.3 ms — the cascade is free at production scale (−1%, i.e.
  timing noise; the gradient dominates). Production-size (8³) overhead can
  be read off a future F1accs run if ever needed.

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

## Q6 — gates (Q-M4, landed)

All in test_quantum_thermostat.jl, predictions in test/unit/qt_predictions.jl
(the verified linear-response machinery: single-mode occupation integral with
the exact (1+α²) pole ω₀ = ω̃/(1+α²), Γ = αω₀; the classical limit collapses
to kT exactly — the constants-cancellation proof is in that file's header).
F1 shipped-filter PSD certificate over τ ∈ {0.005, 0.0152, 0.05, 0.1};
F2 impulse response ≡ closed-form |H_d|² (150k taps, rtol 1e-5);
F3/F4 shipped-filter Lyapunov pins + seeded stream variance (4σ);
F5 determinism/bitwise set (the Q-M1–Q-M3 gates, now running the real
filter). G1 Larmor occupation vs the α-broadened integral of the SHIPPED PSD
(3σ + 1.5% systematic; α ∈ {0.1, 0.5} — the α-DEPENDENCE is asserted,
replacing the classical α-independence gate); G2 Einstein specific heat via
two-T finite difference (c_meas < 0.6 kB; Einstein ≈ 0.50 at x₀ = 3,
classical exactly 1); G3 classical recovery at x₀ ≈ 0.02
(prediction-anchored 3σ + the quadrature assertion E_pred/kT > 0.97 —
θ(0.02) alone is −1%, so "kT to 0.5%" is not a valid statistical gate);
G4 dimer two-mode occupations (mode-resolved; the 2√3 SALC normalization is
measured via _dimer_J, never assumed); G5 MC-mismatch tripwire (> 5σ;
predicted quantum deficit ≈ 0.68 kT at x_o ≈ 2). Verification record behind
the predictions: classical kT anchors at ≤ 2σ over 6 parameter combos,
deterministic dimer mode frequencies to 1e-4, dt-halving bias bound 0.4%.

G6 (example tier, landed 2026-07-19): ring S(q,ω) intensities — one quantum
sLLG spectrum of the 4-ring + field (modes at x = ħω/kT ≈ 1.0/2.2/3.5)
gates the ω-INTEGRATED inelastic S⁺⁻(q_m) per mode against the same
linear-response machinery via `_qt_predict_ring` (unitary mode transform ⇒
`⟨|c_m|²⟩ = 2E_m/b_m`, which is both the equal-time |F₊(q_m)|² observable
and the S(q,ω) bin sum — Parseval): equal-time route 3σ + 4%, S(q,ω) route
3σ + 5%, peak at ω_m = p·b_m within 5%, and the dispersion-wide suppression
assertion (measured I₂/I₀ ≈ 0.066 vs classical b₀/b₂ ≈ 0.286 — every mode
would carry kT classically). Measured deviations ~1%, errors ~1.5% (2¹⁷
frames). Companion example: `examples/bloch_mt.jl` — M(T) of a cubic
ferromagnet, classical linear deficit vs quantum low-T flattening (the
discrete-grid Bloch-law signature), against in-script LSWT estimates.
