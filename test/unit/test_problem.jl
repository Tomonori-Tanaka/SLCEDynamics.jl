# LLGProblem construction: parameter resolution, validation, the resolved
# prefactor/Zeeman fields, and the dynamical energy.

@testset "LLGProblem" begin
    H = MC.TiledHamiltonian(_dimer_model(); dims = (1, 1, 1))
    rng = MersenneTwister(5)

    @testset "scalar / per-atom / per-site parameter resolution" begin
        p1 = LLGProblem(H; magmom = 2.0)
        @test p1.magmom == fill(2.0, 4)
        @test p1.alpha == zeros(4)
        @test p1.g == fill(2.0, 4)
        # per-atom (== per-site here, dims = (1,1,1)) vector accepted
        p2 = LLGProblem(H; magmom = [2.0, 3.0, 1.0, 1.0])
        @test p2.magmom == [2.0, 3.0, 1.0, 1.0]
        # tiling to a 2× supercell: per-atom entries repeat via site_atom
        H2 = MC.TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))
        p3 = LLGProblem(H2; magmom = [2.0, 3.0, 1.0, 1.0])
        for s = 1:MC.n_sites(H2)
            @test p3.magmom[s] == [2.0, 3.0, 1.0, 1.0][MC.site_atom(H2, s)]
        end
    end

    @testset "prefactor and Zeeman gradient" begin
        prob = LLGProblem(H; magmom = 2.0, alpha = 0.5, g = 2.0,
                          b_ext = (0.0, 0.0, 3.0))
        for s = 1:4
            if H.site_active[s]
                @test prob.pref[s] ≈ 2.0 / (SD.HBAR_EV_FS * 2.0 * 1.25)
                @test prob.gzee[s] ≈ SVector(0.0, 0.0, -2.0 * SD.MU_B_EV_T * 3.0)
            else
                @test prob.pref[s] == 0.0
                @test prob.gzee[s] == zero(SVector{3,Float64})
            end
        end
    end

    @testset "inactive-site magmom is ignored" begin
        @test any(!, H.site_active)
        mm = [2.0, 2.0, 2.0, 2.0]
        mm[findfirst(!, H.site_active)] = 0.0        # invalid, but inactive
        @test LLGProblem(H; magmom = mm) isa LLGProblem
    end

    @testset "validation" begin
        @test_throws ArgumentError LLGProblem(H; magmom = 0.0)
        @test_throws ArgumentError LLGProblem(H; magmom = -1.0)
        @test_throws ArgumentError LLGProblem(H; magmom = 2.0, alpha = -0.1)
        @test_throws ArgumentError LLGProblem(H; magmom = 2.0, g = 0.0)
        @test_throws ArgumentError LLGProblem(H; magmom = [1.0, 2.0, 3.0])
        @test_throws ArgumentError LLGProblem(H; magmom = 2.0,
                                              b_ext = (NaN, 0.0, 0.0))
        @test_throws ArgumentError LLGProblem(H; magmom = "two")
    end

    @testset "dynamical energy = SLCE + Zeeman over active sites" begin
        prob = LLGProblem(H; magmom = 2.0, b_ext = (0.0, 0.0, 3.0))
        config = _rand_config(rng, H)
        ez = sum(-2.0 * SD.MU_B_EV_T * 3.0 * config[s][3]
                 for s = 1:4 if H.site_active[s])
        @test total_energy(prob, config) ≈ total_energy(H, config) + ez atol = 1e-15
    end

    @testset "show reports active-site ranges only" begin
        prob = LLGProblem(H; magmom = 2.0)
        @test occursin("LLGProblem", sprint(show, prob))
        # a garbage placeholder on an inactive site must not leak into the range
        mm = [2.0, 2.0, 2.0, 2.0]
        mm[findfirst(!, H.site_active)] = 0.0
        s = sprint(show, LLGProblem(H; magmom = mm))
        @test occursin("magmom ∈ [2.0, 2.0]", s)
    end
end
