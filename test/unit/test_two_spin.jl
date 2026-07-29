# Validation gate (c): the two-spin Heisenberg dimer E = J e₁·e₂ at α = 0. The
# total spin s = e₁ + e₂ is conserved and each spin rotates uniformly about it
# with the constant rotation vector ω = −p·J·s, p = g/(ħ·magmom) — an analytic
# trajectory, compared pointwise. J < 0 (ferro) precesses about +s.

@testset "two-spin Heisenberg dimer (gate c)" begin
    model = _dimer_model(-0.02)
    H = MC.TiledHamiltonian(model; dims = (1, 1, 1))
    J = _dimer_J(H)
    # the isotropic l = 1 SALC carries the 2√3 normalization (the oracle's
    # Heisenberg pin): E = jϕ·2√3·(e₁·e₂)
    @test J ≈ -0.02 * 2 * sqrt(3) rtol = 1e-12

    θ = 0.4
    up = SVector(0.0, 0.0, 1.0)
    e1 = SVector(sin(θ), 0.0, cos(θ))
    e2 = SVector(-sin(θ), 0.0, cos(θ))
    config0 = MC.SpinConfig([e1, e2, up, up])
    szlen = 2 * cos(θ)                                  # |s|, s ∥ +z

    _omega_z(magmom, g = 2.0) = -g * J * szlen / (SD.HBAR_EV_FS * magmom)
    _analytic1(Ω, t) = SVector(sin(θ) * cos(Ω * t), sin(θ) * sin(Ω * t), cos(θ))

    @testset "pointwise trajectory, invariants" begin
        prob = LLGProblem(H; magmom = 2.0)
        Ω = _omega_z(2.0)                               # ≈ 0.194 rad/fs, T ≈ 32 fs
        @test Ω > 0                                     # ferro: rotation about +s
        nsteps = 6000
        dt = 0.005
        res = run_llg(prob, config0; dt = dt, nsteps = nsteps, renorm_interval = 0)
        t = nsteps * dt                                 # ≈ one period
        # rtol 1e-5 has ~2× margin over the O(dt²) error at this dt (the thinnest
        # gate in the suite) — if it ever trips, halve dt before suspecting physics
        @test res.config[1] ≈ _analytic1(Ω, t) rtol = 1e-5
        @test res.config[2] ≈ SVector(-1.0, -1.0, 1.0) .* _analytic1(Ω, t) rtol = 1e-5
        # ...and the halve-dt diagnosis is built in: the same endpoint at dt/2
        # must sit ≳3× closer to the closed form (O(dt²) ⇒ ~4×), so a trip above
        # self-separates "resolution" from "physics" without re-deriving anything
        res2 = run_llg(prob, config0; dt = dt / 2, nsteps = 2 * nsteps,
                       renorm_interval = 0)
        err1 = norm(res.config[1] - _analytic1(Ω, t))
        err2 = norm(res2.config[1] - _analytic1(Ω, t))
        @test err2 < err1 / 3
        # invariants: total spin, angle, energy (conserved to integrator order)
        s_end = res.config[1] + res.config[2]
        @test s_end ≈ SVector(0.0, 0.0, szlen) atol = 1e-6
        @test dot(res.config[1], res.config[2]) ≈ dot(e1, e2) atol = 1e-6
        @test res.energies[end] ≈ res.energies[1] atol = 1e-7
    end

    @testset "magmom = 2 → 4 halves the frequency" begin
        prob4 = LLGProblem(H; magmom = 4.0)
        Ω4 = _omega_z(4.0)
        @test Ω4 ≈ _omega_z(2.0) / 2
        res = run_llg(prob4, config0; dt = 0.005, nsteps = 6000)
        @test res.config[1] ≈ _analytic1(Ω4, 6000 * 0.005) rtol = 1e-5
    end

    @testset "uniform B_ext ∥ s adds the Larmor rate" begin
        B = 5.0
        prob = LLGProblem(H; magmom = 2.0, b_ext = (0.0, 0.0, B))
        Ω = _omega_z(2.0) + _larmor_omega(2.0, B)
        res = run_llg(prob, config0; dt = 0.005, nsteps = 6000)
        @test res.config[1] ≈ _analytic1(Ω, 6000 * 0.005) rtol = 1e-5
    end

    @testset "α > 0 relaxes toward alignment (ferro ground state)" begin
        prob = LLGProblem(H; magmom = 2.0, alpha = 1.0)
        res = run_llg(prob, config0; dt = 0.1, nsteps = 4000)
        @test dot(res.config[1], res.config[2]) > 0.999   # e₁·e₂ → 1
        @test res.energies[end] < res.energies[1]
        @test res.energies[end] ≈ J atol = 1e-4           # E → J at alignment
    end
end
