# S(q,ω) core plumbing: the FFT kernel vs a reference DFT, q-space geometry
# (ordering pin against the upstream supercell contract, translation covariance,
# q_path snapping), the exact Parseval sum rules, the Welch/ensemble layer, and
# bit-determinism. The analytic physics gates live in test_sqw_gates.jl.

# Reference O(n²) DFT — the independent implementation the FFT is gated against.
_ref_dft(x::Vector{ComplexF64}) =
    ComplexF64[sum(x[n+1] * cis(-2π * k * n / length(x))
                   for n = 0:length(x)-1) for k = 0:length(x)-1]

# fftshifted index of the signed bin k′.
_iw(kp::Int, M::Int) = kp + (M >>> 1) + 1

@testset "S(q,ω) core" begin
    @testset "radix-2 FFT ≡ reference DFT" begin
        rng = MersenneTwister(11)
        for n in (2, 4, 8, 32, 128, 512)
            x = ComplexF64[randn(rng) + im * randn(rng) for _ = 1:n]
            a = copy(x)
            SD._fft_pow2!(a, SD._Twiddle(n))
            @test maximum(abs.(a - _ref_dft(x))) <= 1e-12 * n
            # Parseval
            @test sum(abs2, a) ≈ n * sum(abs2, x) rtol = 1e-12
        end
        @test_throws ArgumentError SD._Twiddle(3)
        @test_throws ArgumentError SD._Twiddle(1)
        # window power: periodic Hann has W₂ = 3/8 exactly
        w = Vector{Float64}(undef, 64)
        SD._fill_window!(w, :hann)
        @test SD._window_power(w) ≈ 3 / 8 rtol = 1e-14
        SD._fill_window!(w, :none)
        @test SD._window_power(w) == 1.0
        @test_throws ArgumentError SD._fill_window!(w, :hamming)
        # two-sided axis: ascending, Nyquist only at −M/2
        om = SD._freq_axis(8, 0.5)
        @test om == [2π * k / 4 for k = -4:3]
    end

    # a small thermal workhorse: the biquadratic 2-atom model on a 2×1×1 supercell
    H2 = MC.TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))
    cr2 = _biquadratic_crystal()
    prob2 = LLGProblem(H2; magmom = 2.0, alpha = 0.8)
    c0 = _rand_config(MersenneTwister(4), H2)
    res2 = run_llg(prob2, c0; dt = 0.05, nsteps = 640, kT = 0.02, seed = 21,
                   measure_interval = 5,
                   observables = [trajectory_observable(H2)])
    tr2 = trajectory(res2)
    qs2 = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 2.0]]

    @testset "ordering pin: kernel phases ≡ supercell Cartesian positions" begin
        sup = supercell_crystal(cr2, (2, 1, 1))
        rcart = cartesian_positions(sup)
        for q in qs2
            f = SVector{3,Float64}(q)
            m = SD._q_ints(f, (2, 1, 1))
            phi = Vector{ComplexF64}(undef, MC.n_sites(H2))
            SD._fill_phases!(phi, zeros(H2.n_cell_atoms), H2, cr2, m)
            qc = SD._q_cartesian(cr2, f)
            for s = 1:MC.n_sites(H2)
                @test phi[s] ≈ cis(-dot(qc, @view rcart[:, s])) atol = 1e-12
            end
        end
    end

    @testset "Parseval sum rules (:none and :hann), static consistency" begin
        for window in (:none, :hann)
            r = structure_factor(tr2.traj, tr2.times, H2, cr2, qs2;
                                 window = window)
            M = r.nfft
            w = Vector{Float64}(undef, M)
            SD._fill_window!(w, window)
            W2 = SD._window_power(w)
            # brute-force s^α(q,t) through the Cartesian route (independent
            # path). The estimator's mean is over the ANALYZED span — the samples
            # the segments transform, `(nsegments−1)·hop + M`, which is exactly M
            # here (nsegments = 1) — never the full trimmed window (review
            # 2026-08-11 M6: a full-window mean leaked residual DC into ω = 0).
            sup = supercell_crystal(cr2, (2, 1, 1))
            rcart = cartesian_positions(sup)
            n = MC.n_sites(H2)
            spin_means = [sum(tr2.traj[α, s, j] for j = 1:M) / M for α = 1:3, s = 1:n]
            for (iq, q) in enumerate(qs2)
                qc = SD._q_cartesian(cr2, SVector{3,Float64}(q))
                for α = 1:3
                    sqt = [sum(cis(-dot(qc, @view rcart[:, s])) *
                               (tr2.traj[α, s, j] - spin_means[α, s]) for s = 1:n) /
                           sqrt(n) for j = 1:M]
                    lhs = sum(real(r.S[α, α, iq, iw]) for iw = 1:M) /
                          (M * r.dt_meas)
                    rhs = sum(w[j]^2 * abs2(sqt[j]) for j = 1:M) / (M * W2)
                    @test lhs ≈ rhs rtol = 1e-10
                end
            end
        end
    end

    @testset "channel-level global sum rule" begin
        sr = SD.channel_sumrule(tr2.traj, tr2.times, H2, cr2)
        @test sr.lhs ≈ sr.rhs rtol = 1e-10
    end

    @testset "translation covariance (spatial-sign pin, generic phase)" begin
        # cyclically shift every spin by one cell along a₁ on the 2×1×1 torus:
        # s(q) picks up e^{−2πi f·c₀}; S is invariant
        n = MC.n_sites(H2)
        n_a = H2.n_cell_atoms
        perm = [mod1(s + n_a, n) for s = 1:n]          # site of the shifted spin
        shifted = tr2.traj[:, perm, :]
        r0 = structure_factor(tr2.traj, tr2.times, H2, cr2, qs2; window = :none)
        r1 = structure_factor(shifted, tr2.times, H2, cr2, qs2; window = :none)
        @test maximum(abs.(r1.S - r0.S)) <= 1e-10 * (1 + maximum(abs.(r0.S)))
        @test maximum(abs.(r1.S_el - r0.S_el)) <= 1e-12
    end

    @testset "q_path" begin
        p = q_path(cr2, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 1.0]];
                   npoints = 9, dims = (2, 1, 1))
        @test all(q -> all(abs.(q .* (2, 1, 1) - round.(q .* (2, 1, 1))) .<
                           1e-12), p.qs)
        @test allunique(p.qs) || all(p.qs[i] != p.qs[i+1]
                                     for i = 1:length(p.qs)-1)
        @test issorted(p.x) && p.x[1] == 0.0
        @test p.qs[p.vert_idx[1]] == SVector(0.0, 0.0, 0.0)
        @test p.qs[p.vert_idx[end]] == SVector(0.5, 0.0, 1.0)
        @test length(p.qs) == length(p.qs_requested) == length(p.x)
        # without dims: no snapping, and the compute layer rejects incommensurate q
        praw = q_path(cr2, [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]; npoints = 4)
        @test_throws ArgumentError structure_factor(tr2.traj, tr2.times, H2,
                                                    cr2, praw.qs)
        @test_throws ArgumentError q_path(cr2, [[0.0, 0.0, 0.0]])
        @test_throws ArgumentError q_path(cr2, [[0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0]]; npoints = 5)
    end

    @testset "Welch segmenting and parameter derivation" begin
        r = structure_factor(tr2.traj, tr2.times, H2, cr2, qs2;
                             nsegments = 3, overlap = 0.5, discard = 9)
        # L = 129 − 9 = 120; M = 32 gives (3−1)·16 + 32 = 64 ≤ 120; M = 64 needs
        # 128 > 120 — so the derived power-of-two segment length is 32
        @test r.nfft == 32 && r.nsegments == 3 && r.discard == 9
        @test length(r.omegas) == 32
        r2 = structure_factor(tr2.traj, tr2.times, H2, cr2, qs2;
                              nsegments = 3, overlap = 0.5, seglength = 16)
        @test r2.nfft == 16
        @test_throws ArgumentError structure_factor(tr2.traj, tr2.times, H2,
                                                    cr2, qs2; seglength = 20)
        @test_throws ArgumentError structure_factor(tr2.traj, tr2.times, H2,
                                                    cr2, qs2; nsegments = 200)
        @test_throws ArgumentError structure_factor(tr2.traj, tr2.times, H2,
                                                    cr2, qs2; overlap = 1.0)
        @test_throws ArgumentError structure_factor(tr2.traj, tr2.times, H2,
                                                    cr2, qs2; discard = 128)
    end

    # The recorded series is an ON-DISK format promise: a checkpoint written by an older
    # version is read by a newer one. Nothing else pins its layout as a literal — the
    # file round-trip (`structure_factor(path,…) == structure_factor(result,…)` below) moves
    # writer and reader together, so it passes unchanged if the layout is transposed, and
    # the analytic S(q,ω) gates catch a reordering only indirectly. Assert the layout
    # against the CONFIGURATION itself: xyz fastest, then site.
    @testset "series layout is xyz-fastest-then-site (stored-file contract)" begin
        # The measurement grid includes step 0, so `nsteps == measure_interval` gives
        # exactly two columns whose contents are known independently of the writer: the
        # initial configuration (door-projected — the state the run starts from)
        # and the returned one.
        r0 = run_llg(prob2, c0; dt = 0.05, nsteps = 5, kT = 0.02, seed = 1,
                     measure_interval = 5, observables = [trajectory_observable(H2)])
        c0p = SD._config_projected(c0, H2.site_active)
        ser = r0.series[:spins]
        @test size(ser) == (3 * MC.n_sites(H2), 2)
        @test r0.times == [0.0, 5 * 0.05]
        @test ser[:, 1] == vec(MC.to_matrix(c0p))
        @test ser[:, end] == vec(MC.to_matrix(r0.config))
        @test ser[:, 1] != ser[:, end]          # the run moved, so this is not vacuous
        # spelled out, so a transpose cannot satisfy it by coincidence
        for s = 1:min(3, MC.n_sites(H2)), k = 1:3
            @test ser[3*(s-1)+k, 1] == c0p[s][k]
        end
    end

    @testset "seed-ensemble averaging" begin
        seeds = (31, 32, 33)
        runs = [run_llg(prob2, c0; dt = 0.05, nsteps = 640, kT = 0.02,
                        seed = sd, measure_interval = 5,
                        observables = [trajectory_observable(H2)])
                for sd in seeds]
        re = structure_factor(runs, H2, cr2, qs2)
        @test re.nrealizations == 3 && re.err !== nothing
        @test all(re.err .>= 0)
        singles = [structure_factor(r, H2, cr2, qs2) for r in runs]
        @test re.S ≈ sum(s.S for s in singles) ./ 3 rtol = 1e-12
        r2 = structure_factor(runs[1:2], H2, cr2, qs2)
        @test r2.nrealizations == 2 && r2.err === nothing
        # mismatched time grid is rejected
        short = run_llg(prob2, c0; dt = 0.05, nsteps = 320, kT = 0.02,
                        seed = 34, measure_interval = 5,
                        observables = [trajectory_observable(H2)])
        @test_throws ArgumentError structure_factor([runs[1], short], H2, cr2,
                                                    qs2)
    end

    @testset "determinism and the checkpoint route" begin
        r1 = structure_factor(tr2.traj, tr2.times, H2, cr2, qs2; ntasks = 1)
        r3 = structure_factor(tr2.traj, tr2.times, H2, cr2, qs2; ntasks = 3)
        @test r1.S == r3.S && r1.S_el == r3.S_el
        @test structure_factor(tr2.traj, tr2.times, H2, cr2, qs2).S == r1.S
        dir = mktempdir()
        path = joinpath(dir, "traj.jld2")
        rck = run_llg(prob2, c0; dt = 0.05, nsteps = 640, kT = 0.02, seed = 21,
                      measure_interval = 5,
                      observables = [trajectory_observable(H2)],
                      checkpoint = path)
        rf = structure_factor(path, H2, cr2, qs2; window = :hann)
        rm = structure_factor(rck, H2, cr2, qs2; window = :hann)
        @test rf.S == rm.S && rf.S_el == rm.S_el
        @test_throws ArgumentError structure_factor(joinpath(dir, "no.jld2"),
                                                    H2, cr2, qs2)
    end

    @testset "off-grid final measurement is dropped" begin
        ro = run_llg(prob2, c0; dt = 0.05, nsteps = 642, kT = 0.02, seed = 21,
                     measure_interval = 5,
                     observables = [trajectory_observable(H2)])
        tro = trajectory(ro)
        @test length(tro.times) == 129            # 130th (off-grid) col dropped
        @test tro.times == collect(0.0:0.25:32.0)
        # raw method on the unpruned series rejects the non-uniform grid
        raw = reshape(ro.series[:spins], 3, MC.n_sites(H2), :)
        @test_throws ArgumentError structure_factor(raw, ro.times, H2, cr2, qs2)
    end

    @testset "validation" begin
        @test_throws ArgumentError trajectory(res2; name = :missing)
        @test_throws DimensionMismatch structure_factor(tr2.traj[:, 1:2, :],
                                                        tr2.times, H2, cr2, qs2)
        @test_throws ArgumentError structure_factor(tr2.traj, tr2.times, H2,
                                                    cr2, [[0.3, 0.0, 0.0]])
        @test_throws ArgumentError structure_factor(tr2.traj, tr2.times, H2,
                                                    cr2, Vector{Float64}[])
        @test_throws ArgumentError structure_factor(tr2.traj, tr2.times, H2,
                                                    cr2, qs2; ntasks = 0)
        # wrong crystal (atom count) is rejected
        @test_throws ArgumentError structure_factor(tr2.traj, tr2.times, H2,
                                                    _dimer_crystal(), qs2)
    end

    @testset "temporal-aliasing screen (audit #8)" begin
        # Synthetic m = 1 spiral on the 4-site ring, built directly on the analysis
        # grid (dtm = 1): e_j(t) precesses at a hand-chosen ω, so the sampled
        # spectrum is one exact bin. M = 256 ⇒ Nyquist bin 128.
        Hr = MC.TiledHamiltonian(_ring_model(-0.02); dims = (1, 1, 1))
        crr = _dimer_crystal()
        M = 256
        times = collect(0.0:1.0:(M-1))
        ε = 0.05
        spiral(ω) = begin
            traj = Array{Float64,3}(undef, 3, 4, M)
            for k = 1:M, j = 1:4
                φ = 2π * (j - 1) / 4 + ω * (k - 1)
                traj[:, j, k] .= (ε * cos(φ), ε * sin(φ), sqrt(1 - ε^2))
            end
            traj
        end
        q1 = [[0.0, 0.0, 1.0]]
        # healthy: ω at bin 38 (0.297·π) — far below the Nyquist, no warning
        @test_logs structure_factor(spiral(2π * 38 / M), times, Hr, crr, q1;
                                    window = :none)
        # aliased: ω at "bin 131" > 128 folds to −125, i.e. |ω| = 0.977·π — inside
        # the top-5 % edge band, so the screen fires
        @test_logs (:warn, r"folds back") match_mode = :any structure_factor(
            spiral(2π * 131 / M), times, Hr, crr, q1; window = :none)
    end

    @testset "samples beyond the analyzed span do not touch the spectrum (review M6)" begin
        # With nt = 24, nsegments = 1: M = prevpow(2, 24) = 16, so samples 17–24 are
        # never transformed. The per-site mean (and hence S_el and the subtracted
        # DC) must be computed over the 16 analyzed samples only — before the fix
        # the mean ran over all 24, and a drifting tail leaked a spurious residual-DC
        # weight into the INELASTIC ω = 0 bin. The gate is an invariance: replacing
        # the never-transformed tail with junk must change nothing.
        Hr = MC.TiledHamiltonian(_ring_model(-0.02); dims = (1, 1, 1))
        crr = _dimer_crystal()
        nt = 24
        times = collect(0.0:1.0:(nt-1))
        rng = MersenneTwister(5)
        traj = Array{Float64,3}(undef, 3, 4, nt)
        for k = 1:nt, j = 1:4
            v = randn(rng, 3)
            traj[:, j, k] .= v ./ norm(v)
        end
        q1 = [[0.0, 0.0, 1.0]]
        r1 = structure_factor(traj, times, Hr, crr, q1; window = :none)
        traj2 = copy(traj)
        for k = 17:nt, j = 1:4                       # junk tail, unit columns
            traj2[:, j, k] .= (0.0, 0.0, (-1.0)^k)
        end
        r2 = structure_factor(traj2, times, Hr, crr, q1; window = :none)
        @test r2.S == r1.S
        @test r2.S_el == r1.S_el
    end
end
