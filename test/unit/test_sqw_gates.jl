# S(q,ω) analytic physics gates. All deterministic (α = 0, Depondt, :none
# window) with COMMENSURATE sampling: dt is chosen so the analytic mode
# frequency satisfies Ω·(mi·dt)·M = 2π·k₀, concentrating the DFT on exact bins
# with closed-form values. These gates freeze the temporal sign (positive
# precession about +ẑ → +ω), the spatial sign (+q vs −q), the √N normalization,
# and the ω axis — do not touch sqw.jl's conventions without re-running them.

_iwbin(kp::Int, M::Int) = kp + (M >>> 1) + 1

@testset "S(q,ω) analytic gates" begin
    @testset "Larmor (gate a): closed-form single-bin spectrum" begin
        # Depondt is exact for a constant field, so the sampled trajectory is
        # exactly (cos ω_L t_n, sin ω_L t_n, 0) — the DFT has closed forms.
        H = MC.TiledHamiltonian(_uniaxial_model(1e-20); dims = (1, 1, 1))
        cru = _uniaxial_crystal()
        B = 10.0
        ωl = _larmor_omega(2.0, B)
        M, k0 = 256, 3
        dts = 2π * k0 / (ωl * M)                   # mi = 1 ⇒ Δt_s = dt
        c0 = MC.SpinConfig([SVector(1.0, 0.0, 0.0)])
        prob = LLGProblem(H; magmom = 2.0, b_ext = (0.0, 0.0, B))
        result = run_llg(prob, c0; dt = dts, nsteps = M - 1, measure_interval = 1,
                      renorm_interval = 0,
                      observables = [trajectory_observable(H)])
        r = structure_factor(result, H, cru, [[0.0, 0.0, 0.0]]; window = :none)
        @test r.nfft == M
        MΔ = M * r.dt_meas
        sxx = [real(r.S[1, 1, 1, iw]) for iw = 1:M]
        @test sxx[_iwbin(k0, M)] ≈ MΔ / 4 rtol = 1e-8
        @test sxx[_iwbin(-k0, M)] ≈ MΔ / 4 rtol = 1e-8
        offpeak = [sxx[iw] for iw = 1:M if iw != _iwbin(k0, M) &&
                   iw != _iwbin(-k0, M)]
        @test maximum(offpeak) <= 1e-10 * MΔ
        @test sum(sxx) / MΔ ≈ 0.5 rtol = 1e-10        # bin-sum rule (N = 1)
        # temporal-sign pin: ALL S^{+−} weight at +ω_{k₀}
        spm = sqw_plusminus(r)
        @test spm[1, _iwbin(k0, M)] ≈ MΔ rtol = 1e-8
        @test spm[1, _iwbin(-k0, M)] <= 1e-10 * MΔ
        # the ω axis: the peak sits at +ω_L on the physical axis
        @test r.omegas[_iwbin(k0, M)] ≈ ωl rtol = 1e-12
        @test r.energies_mev == 1000 .* HBAR_EV_FS .* r.omegas
        # no elastic weight (the time mean vanishes on the commensurate window)
        @test maximum(abs.(r.S_el)) <= 1e-12
        # :perp is NaN at Γ by convention
        @test all(isnan, sqw_perp(r)[1, :])
        # magmom does not move a pure-Zeeman peak
        prob5 = LLGProblem(H; magmom = 5.0, b_ext = (0.0, 0.0, B))
        res5 = run_llg(prob5, c0; dt = dts, nsteps = M - 1,
                       measure_interval = 1, renorm_interval = 0,
                       observables = [trajectory_observable(H)])
        r5 = structure_factor(res5, H, cru, [[0.0, 0.0, 0.0]]; window = :none)
        @test argmax(sqw_plusminus(r5)[1, :]) == _iwbin(k0, M)
    end

    @testset "dimer (gate b): conserved q = 0, one mode at q = (0,0,2)" begin
        H = MC.TiledHamiltonian(_dimer_model(-0.02); dims = (1, 1, 1))
        crd = _dimer_crystal()
        Jp = _dimer_J(H)                            # measured (SALC 2√3 factor)
        @test Jp < 0
        θ = 0.3
        magmom = 2.0
        p = 2.0 / (SD.HBAR_EV_FS * magmom)
        # the symmetric cant rotates rigidly about ẑ — an EXACT solution with
        # Ω = 2p|J|cosθ (any θ), so only the O(dt²) integrator phase drift limits
        # the gate; ω·dt ≈ 0.01 keeps the peak inside one bin
        Ω = 2 * p * abs(Jp) * cos(θ)
        M, k0, mi = 512, 16, 20
        dts = 2π * k0 / (Ω * M)
        dt = dts / mi
        up = SVector(0.0, 0.0, 1.0)
        c0 = MC.SpinConfig([SVector(sin(θ), 0.0, cos(θ)),
                            SVector(-sin(θ), 0.0, cos(θ)), up, up])
        prob = LLGProblem(H; magmom = magmom)
        result = run_llg(prob, c0; dt = dt, nsteps = mi * (M - 1),
                      measure_interval = mi, renorm_interval = 0,
                      observables = [trajectory_observable(H)])
        qs = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
        r = structure_factor(result, H, crd, qs; window = :none)
        @test r.nfft == M
        MΔ = M * r.dt_meas
        # (i) total-spin conservation: the q = 0 channel is purely elastic
        tr0 = sqw_trace(r)
        @test maximum(tr0[1, :]) <= 1e-10 * MΔ
        # rtol 1e-5: e_z is conserved by the flow but drifts O(dt²) over the
        # 10k discrete steps (measured ≈ 1e-6 relative)
        @test real(r.S_el[3, 3, 1]) ≈ 2 * cos(θ)^2 rtol = 1e-5
        # (ii) all inelastic weight in the +ω_{k₀} bin of S^{+−} at (0,0,2)
        spm = sqw_plusminus(r)
        @test argmax(spm[2, :]) == _iwbin(k0, M)
        @test spm[2, _iwbin(k0, M)] ≈ 2 * sin(θ)^2 * MΔ rtol = 1e-3
        @test spm[2, _iwbin(k0, M)] / sum(spm[2, :]) >= 0.99
        @test spm[2, _iwbin(-k0, M)] <= 1e-6 * spm[2, _iwbin(k0, M)]
        # (iii) sum rules
        @test sum(spm[2, :]) / MΔ ≈ 2 * sin(θ)^2 rtol = 1e-4
        trace = trajectory(result)
        sr = SD.channel_sumrule(trace.traj, trace.times, H, crd)
        @test sr.lhs ≈ sr.rhs rtol = 1e-10
        @test sr.rhs ≈ 2 * sin(θ)^2 rtol = 1e-4
        # inactive atoms are excluded: randomizing their frozen directions
        # changes nothing, bitwise
        rng = MersenneTwister(9)
        c0r = MC.SpinConfig([c0[1], c0[2], _rand_spin(rng), _rand_spin(rng)])
        resr = run_llg(prob, c0r; dt = dt, nsteps = mi * (M - 1),
                       measure_interval = mi, renorm_interval = 0,
                       observables = [trajectory_observable(H)])
        rr = structure_factor(resr, H, crd, qs; window = :none)
        @test rr.S == r.S && rr.S_el == r.S_el
    end

    @testset "ring (gate d): dispersion, spatial sign, magmom scaling" begin
        H = MC.TiledHamiltonian(_ring_model(-0.02); dims = (1, 1, 1))
        crd = _dimer_crystal()
        Jr = _ring_J(H)                             # measured
        @test Jr < 0
        @test all(H.site_active)
        ε = 0.01
        magmom = 2.0
        p = 2.0 / (SD.HBAR_EV_FS * magmom)
        # the m = 1 single-magnon spiral is an exact solution:
        # ω = 2p|J|·cosθ·(1 − cos(2π/4)), cosθ = √(1 − ε²)
        ω1 = 2 * p * abs(Jr) * sqrt(1 - ε^2)
        M, k0, mi = 1024, 16, 10
        dts = 2π * k0 / (ω1 * M)
        dt = dts / mi
        c0 = MC.SpinConfig([SVector(ε * cospi(2 * (j - 1) / 4),
                                    ε * sinpi(2 * (j - 1) / 4),
                                    sqrt(1 - ε^2)) for j = 1:4])
        prob = LLGProblem(H; magmom = magmom)
        result = run_llg(prob, c0; dt = dt, nsteps = mi * (M - 1),
                      measure_interval = mi, renorm_interval = 0,
                      observables = [trajectory_observable(H)])
        qs = [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]]
        r = structure_factor(result, H, crd, qs; window = :none)
        spm = sqw_plusminus(r)
        # the mode sits at (+q, +ω) — both signs pinned at once
        @test argmax(spm[1, :]) == _iwbin(k0, M)
        @test spm[1, _iwbin(k0, M)] / sum(spm[1, :]) >= 0.95
        @test r.omegas[_iwbin(k0, M)] ≈ ω1 rtol = 1e-12   # axis, not dynamics
        # the −q partner (0,0,3) ≡ (0,0,−1) carries (almost) nothing
        @test sum(spm[2, :]) <= 1e-4 * sum(spm[1, :])
        # magmom = 4 halves every frequency: same dt, the peak moves to k₀/2
        prob4 = LLGProblem(H; magmom = 4.0)
        res4 = run_llg(prob4, c0; dt = dt, nsteps = mi * (M - 1),
                       measure_interval = mi, renorm_interval = 0,
                       observables = [trajectory_observable(H)])
        r4 = structure_factor(res4, H, crd, qs; window = :none)
        @test argmax(sqw_plusminus(r4)[1, :]) == _iwbin(k0 ÷ 2, M)
    end
end
