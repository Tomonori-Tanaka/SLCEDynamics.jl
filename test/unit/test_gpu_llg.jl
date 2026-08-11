# GPU LLG gates, all on the KernelAbstractions CPU backend (CI needs no GPU;
# decision record docs/specs/gpu-llg.md). The bitwise tier compares the device
# path against a composite keyed reference — the upstream gradient lane
# reference (`SLCEMonteCarlo._gradient_lane_ref!`) plus the literal host
# `_omega`/`_rotate` stage expressions — never against the CPU `run_llg` (whose
# gradient fold order differs by design). CPU↔GPU comparisons are tolerance
# gates on non-chaotic fixtures.

using KernelAbstractions: KernelAbstractions, CPU
using JLD2: JLD2, jldopen

# Copy one checkpoint entry/group into the v1 rewrite, downgrading the header.
function _copy_ckpt_group!(fout, fin, k)
    if k == "schema_version"
        fout[k] = 1
    elseif k in ("run/compute", "run/backend", "run/workgroupsize")
        # dropped in v1
    elseif fin[k] isa JLD2.Group
        for kk in keys(fin[k])
            _copy_ckpt_group!(fout, fin, k * "/" * kk)
        end
    else
        fout[k] = fin[k]
    end
    return nothing
end

# One host reference step of the device arithmetic (shared draw `gth` feeds both
# stages, exactly like the kernels).
function _ref_gpu_step!(config::MC.SpinConfig, prob::LLGProblem,
                        H::MC.TiledHamiltonian, dtf::Float64, ws::Int,
                        gth::Vector{SVector{3,Float64}},
                        integrator::SD.AbstractIntegrator)
    n = MC.n_sites(H)
    G = Vector{SVector{3,Float64}}(undef, n)
    MC._gradient_lane_ref!(G, H, config, MC._zrows(H, config), ws)
    epred = similar(config)
    omega1 = Vector{SVector{3,Float64}}(undef, n)
    for s = 1:n
        e = config[s]
        if !H.site_active[s]
            epred[s] = e
            continue
        end
        gt = G[s] + prob.gzee[s] + gth[s]
        ω1 = -prob.pref[s] * (gt + prob.alpha[s] * cross(e, gt))
        omega1[s] = ω1
        if integrator isa DepondtMertens
            epred[s] = SD._rotate(e, ω1 * dtf)
        else
            ep = e + dtf * cross(ω1, e)
            epred[s] = ep / norm(ep)
        end
    end
    MC._gradient_lane_ref!(G, H, epred, MC._zrows(H, epred), ws)
    for s = 1:n
        H.site_active[s] || continue
        ep = epred[s]
        gt = G[s] + prob.gzee[s] + gth[s]
        ω2 = -prob.pref[s] * (gt + prob.alpha[s] * cross(ep, gt))
        if integrator isa DepondtMertens
            config[s] = SD._rotate(config[s], (omega1[s] + ω2) * (dtf / 2))
        else
            e = config[s] + (dtf / 2) * (cross(omega1[s], config[s]) +
                                         cross(ω2, ep))
            config[s] = e / norm(e)
        end
    end
    return config
end

# The composite reference loop (noise → step → renorm at the absolute-step
# cadence — the arithmetic contract of _llg_loop_gpu!).
function _ref_gpu_loop(prob, H, c0, dtf, nsteps, ws, integrator;
                       kt = NaN, seed = UInt64(0), renorm_interval = 1000)
    config = copy(c0)
    n = MC.n_sites(H)
    gth = fill(zero(SVector{3,Float64}), n)
    sigma = isfinite(kt) ? SD._sigma_noise(prob, kt, dtf) : zeros(n)
    for step = 1:nsteps
        isfinite(kt) && SD._fill_noise!(gth, H, sigma, seed, step)
        _ref_gpu_step!(config, prob, H, dtf, ws, gth, integrator)
        if renorm_interval > 0 && step % renorm_interval == 0
            SD._renormalize_active!(H, config)
        end
    end
    return config
end

@testset "GPU LLG (KA-CPU gates)" begin
    Hd = MC.TiledHamiltonian(_dimer_model(-0.02); dims = (1, 1, 1))
    Hb = MC.TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))
    gHd = MC.GPUTiledHamiltonian(CPU(), Hd)
    gHb = MC.GPUTiledHamiltonian(CPU(), Hb)
    probd = LLGProblem(Hd; magmom = 2.0, alpha = 0.7)
    probb = LLGProblem(Hb; magmom = 2.0, alpha = 0.5,
                       b_ext = (0.0, 0.0, 2.0))
    rng = MersenneTwister(3)
    c0d = MC.SpinConfig([_rand_spin(rng), _rand_spin(rng), _rand_spin(rng),
                         _rand_spin(rng)])
    c0b = _rand_config(rng, Hb)

    @testset "a1: device noise ≡ host _fill_noise! (bitwise)" begin
        n = MC.n_sites(Hd)
        kt = 0.02
        dtf = 0.05
        sigma = SD._sigma_noise(probd, kt, dtf)
        st = SD.GPULLGState(gHd, probd, c0d, sigma)
        host = fill(zero(SVector{3,Float64}), n)
        kern = SD._noise_kernel!(CPU(), 64)
        for step in (1, 5, 2^33 + 7, 2^40 + 123)
            kern(st.dgth, st.dsigma, st.dactive, UInt64(42), step; ndrange = n)
            KernelAbstractions.synchronize(CPU())
            SD._fill_noise!(host, Hd, sigma, UInt64(42), step)
            @test Vector(st.dgth) == host
            # inactive sites carry an exact +0.0 (never σ·ξ = −0.0)
            @test all(Vector(st.dgth)[s] === zero(SVector{3,Float64})
                      for s = 3:4)
        end
    end

    @testset "a3: gpu_run_llg ≡ composite keyed reference (bitwise)" begin
        for (H, gH, prob, c0) in ((Hd, gHd, probd, c0d), (Hb, gHb, probb, c0b)),
            integ in (DepondtMertens(), HeunProjected()), ws in (4, 32)

            det = SD.gpu_run_llg(prob, c0, gH; dt = 0.05, nsteps = 7,
                                 integrator = integ, measure_interval = 3,
                                 workgroupsize = ws)
            @test det.compute == "gpu:cpu"
            # the reference tests the LOOP arithmetic, so it starts from the
            # state the loop actually starts from — the door-projected config
            c0p = SD._config_projected(c0, H.site_active)
            refc = _ref_gpu_loop(prob, H, c0p, 0.05, 7, ws, integ)
            @test det.config == refc

            th = SD.gpu_run_llg(prob, c0, gH; dt = 0.05, nsteps = 7,
                                integrator = integ, measure_interval = 3,
                                workgroupsize = ws, kT = 0.02, seed = 9)
            refc2 = _ref_gpu_loop(prob, H, c0p, 0.05, 7, ws, integ;
                                  kt = 0.02, seed = UInt64(9))
            @test th.config == refc2
        end
    end

    @testset "a4: repeat/seed/ws sensitivity" begin
        kw = (; dt = 0.05, nsteps = 40, kT = 0.02, measure_interval = 10)
        r1 = SD.gpu_run_llg(probd, c0d, gHd; kw..., seed = 5)
        r2 = SD.gpu_run_llg(probd, c0d, gHd; kw..., seed = 5)
        @test r1.config == r2.config && r1.energies == r2.energies
        r3 = SD.gpu_run_llg(probd, c0d, gHd; kw..., seed = 6)
        @test r3.config != r1.config
        # ws is part of the contract — but only sites with MORE adjacency
        # entries than ws see a different fold grouping, so probe it on the
        # entry-rich biquadratic model (the dimer's few entries per site fold
        # identically for any ws)
        r5 = SD.gpu_run_llg(probb, c0b, gHb; kw..., seed = 5,
                            workgroupsize = 4)
        r6 = SD.gpu_run_llg(probb, c0b, gHb; kw..., seed = 5,
                            workgroupsize = 32)
        @test r5.config != r6.config
    end

    @testset "a5: GPU checkpoint / resume / compute-switch semantics" begin
        dir = mktempdir()
        path = joinpath(dir, "gpu.jld2")
        kw = (; dt = 0.05, nsteps = 20, kT = 0.02, seed = 5,
              measure_interval = 4)
        a = SD.gpu_run_llg(probd, c0d, gHd; kw...)
        b = SD.gpu_run_llg(probd, c0d, gHd; kw..., checkpoint = path,
                           checkpoint_interval = 7)
        @test a.config == b.config && a.energies == b.energies
        # completed-file reconstruction on the device method
        c = resume(path, probd, gHd)
        @test c.config == a.config && c.energies == a.energies
        @test c.compute == "gpu:cpu"
        # extension ≡ uninterrupted device run (absolute-step noise)
        long = SD.gpu_run_llg(probd, c0d, gHd; kw..., nsteps = 40)
        ext = resume(path, probd, gHd; nsteps = 40)
        @test ext.config == long.config && ext.energies == long.energies
        # the CPU resume method refuses a GPU file, loudly
        @test_throws ErrorException resume(path, probd)
        # a CPU file into the GPU method: refused without the explicit opt-in
        cpath = joinpath(dir, "cpu.jld2")
        run_llg(probd, c0d; kw..., checkpoint = cpath)
        @test_throws ErrorException resume(cpath, probd, gHd)
        sw = resume(cpath, probd, gHd; allow_compute_switch = true)
        @test sw.compute == "gpu:cpu" && length(sw.times) == length(a.times)
        # different ws is also a switch
        @test_throws ErrorException resume(path, probd, gHd;
                                           workgroupsize = 32)
        # crash-shaped mid-run GPU file (a throwing observable leaves the last
        # periodic tick on disk) resumes bit-identically on the device
        cpath2 = joinpath(dir, "gpucrash.jld2")
        nm = Ref(0)
        boom = Observable(:energy, 1, v -> begin
                              nm[] += 1
                              nm[] > 4 && error("boom")      # measurement 5
                              v.energy
                          end)
        ref = SD.gpu_run_llg(probd, c0d, gHd; kw...,
                             observables = [Observable(:energy, 1,
                                                       v -> v.energy)])
        @test_throws ErrorException SD.gpu_run_llg(probd, c0d, gHd; kw...,
                                                   observables = [boom],
                                                   checkpoint = cpath2,
                                                   checkpoint_interval = 6)
        cr = resume(cpath2, probd, gHd;
                    observables = [Observable(:energy, 1, v -> v.energy)])
        @test cr.config == ref.config && cr.energies == ref.energies
    end

    @testset "checkpoint schema v1 back-read (compute defaults)" begin
        dir = mktempdir()
        p2 = joinpath(dir, "v2.jld2")
        kw = (; dt = 0.05, nsteps = 12, kT = 0.02, seed = 4,
              measure_interval = 3)
        a = run_llg(probd, c0d; kw..., checkpoint = p2)
        # rewrite as a v1 file: drop the v2 keys, stamp schema 1
        p1 = joinpath(dir, "v1.jld2")
        jldopen(p2, "r") do fin
            jldopen(p1, "w") do fout
                for k in keys(fin)
                    _copy_ckpt_group!(fout, fin, k)
                end
            end
        end
        r = resume(p1, probd; checkpoint = nothing)
        @test r.config == a.config && r.energies == a.energies
        @test r.compute == "cpu"
        # a v1 file into the GPU method: a compute switch (v1 ⇒ "cpu") — refused
        # without the opt-in; with it, the ws fallback (stored 0 → 128) applies
        @test_throws ErrorException resume(p1, probd, gHd; checkpoint = nothing)
        sw = resume(p1, probd, gHd; allow_compute_switch = true,
                    checkpoint = nothing, nsteps = 16)
        @test sw.compute == "gpu:cpu" && sw.nsteps == 16
    end

    @testset "a6: quantum thermostat on the device path" begin
        n = MC.n_sites(Hd)
        kt = 0.02
        dtf = 0.05
        seed = UInt64(77)
        sigma = SD._sigma_noise(probd, kt, dtf)
        # (i) kernel ≡ host _fill_noise_quantum! bitwise across steps with a
        # synthetic 2-section filter — pins the raw cascade arithmetic and the
        # transposed device-state layout independent of the shipped constants;
        # (ii)/(iii) below exercise the real shipped filter end-to-end
        sections = [SD._Biquad(0.8, 0.3, -0.1, -0.5, 0.06),
                    SD._Biquad(1.1, -0.2, 0.05, -0.9, 0.25)]
        noise_filter = SD.ColoredNoiseFilter(sections,
                                             SD._stationary_factor(sections))
        fs_host = SD._init_filter_state(noise_filter, Hd, seed)
        fs_dev = SD._init_filter_state(noise_filter, Hd, seed)
        st = SD.GPULLGState(gHd, probd, c0d, sigma, fs_dev)
        host_gth = fill(zero(SVector{3,Float64}), n)
        qkern = SD._noise_kernel_quantum!(CPU(), 64)
        for step = 1:6
            qkern(st.dgth, st.dxstate, st.dsections, st.dsigma, st.dactive,
                  seed, step; ndrange = n)
            KernelAbstractions.synchronize(CPU())
            SD._fill_noise_quantum!(host_gth, Hd, sigma, fs_host.x, noise_filter,
                                    seed, step)
            @test Vector(st.dgth) == host_gth
            SD._download_filter!(fs_dev, st)
            @test fs_dev.x == fs_host.x
            # inactive: exact +0.0 field, state columns never touched
            @test all(Vector(st.dgth)[s] === zero(SVector{3,Float64})
                      for s = 3:4)
            @test all(all(fs_dev.x[:, s] .=== 0.0) for s = 3:4)
        end

        # (ii) device quantum runs: repeat identity; genuinely differ from
        # classical (the shipped filter reshapes the shared white draws)
        kwq = (; dt = dtf, nsteps = 30, kT = kt, seed = 7,
               measure_interval = 6)
        rc = SD.gpu_run_llg(probd, c0d, gHd; kwq...)
        rq = SD.gpu_run_llg(probd, c0d, gHd; kwq...,
                            thermostat = SD.QuantumThermostat())
        rq2 = SD.gpu_run_llg(probd, c0d, gHd; kwq...,
                             thermostat = SD.QuantumThermostat())
        @test rq.config == rq2.config
        @test rq.config != rc.config
        @test rq.thermostat == "quantum" && rc.thermostat == "classical"

        # (iii) GPU quantum checkpoint / resume / extension bitwise
        dir = mktempdir()
        kwc = (; dt = dtf, kT = kt, seed = 5, measure_interval = 4,
               thermostat = SD.QuantumThermostat())
        path = joinpath(dir, "gpuqt.jld2")
        a = SD.gpu_run_llg(probd, c0d, gHd; kwc..., nsteps = 20)
        b = SD.gpu_run_llg(probd, c0d, gHd; kwc..., nsteps = 20,
                           checkpoint = path, checkpoint_interval = 7)
        @test a.config == b.config && a.energies == b.energies
        c = resume(path, probd, gHd)          # completed → reconstruct
        @test c.config == a.config && c.thermostat == "quantum"
        path2 = joinpath(dir, "gpuqt2.jld2")
        SD.gpu_run_llg(probd, c0d, gHd; kwc..., nsteps = 12,
                       checkpoint = path2)
        long = SD.gpu_run_llg(probd, c0d, gHd; kwc..., nsteps = 36)
        ext = resume(path2, probd, gHd; nsteps = 36)
        @test ext.config == long.config && ext.energies == long.energies
        @test ext.thermostat == "quantum"
        # the CPU method still refuses GPU files (compute check, unchanged)
        @test_throws ErrorException resume(path, probd)
        # GPU quantum validation mirrors the CPU path
        @test_throws ArgumentError SD.gpu_run_llg(probd, c0d, gHd; dt = dtf,
            nsteps = 5, thermostat = SD.QuantumThermostat())
        @test_throws ArgumentError SD.gpu_run_llg(probd, c0d, gHd; dt = 1.0,
            nsteps = 5, kT = 0.5, seed = 1,
            thermostat = SD.QuantumThermostat())
    end

    @testset "b1: Larmor on the device path (Depondt constant-field exact)" begin
        H1 = MC.TiledHamiltonian(_uniaxial_model(1e-20); dims = (1, 1, 1))
        gH1 = MC.GPUTiledHamiltonian(CPU(), H1)
        B = 10.0
        ωl = _larmor_omega(2.0, B)
        prob = LLGProblem(H1; magmom = 2.0, b_ext = (0.0, 0.0, B))
        c0 = MC.SpinConfig([SVector(1.0, 0.0, 0.0)])
        result = SD.gpu_run_llg(prob, c0, gH1; dt = 1.0, nsteps = 3600,
                             renorm_interval = 0)
        t = 3600.0
        @test result.config[1] ≈ SVector(cos(ωl * t), sin(ωl * t), 0.0) atol = 1e-9
    end

    @testset "b2/b4: CPU vs GPU on the non-chaotic dimer (tolerance)" begin
        θ = 0.3
        up = SVector(0.0, 0.0, 1.0)
        c0 = MC.SpinConfig([SVector(sin(θ), 0.0, cos(θ)),
                            SVector(-sin(θ), 0.0, cos(θ)), up, up])
        pd = LLGProblem(Hd; magmom = 2.0)
        det_cpu = run_llg(pd, c0; dt = 0.05, nsteps = 1000,
                          measure_interval = 100)
        det_gpu = SD.gpu_run_llg(pd, c0, gHd; dt = 0.05, nsteps = 1000,
                                 measure_interval = 100)
        # linear roundoff envelope (P1): per-step ULP bias × N steps, generous
        @test maximum(norm.(det_cpu.config .- det_gpu.config)) <= 1e-11
        # same-seed sLLG: identical noise stream, only gradient/rotation ULPs
        th_cpu = run_llg(probd, c0d; dt = 0.05, nsteps = 200, kT = 0.02,
                         seed = 5, measure_interval = 50)
        th_gpu = SD.gpu_run_llg(probd, c0d, gHd; dt = 0.05, nsteps = 200,
                                kT = 0.02, seed = 5, measure_interval = 50)
        @test maximum(norm.(th_cpu.config .- th_gpu.config)) <= 1e-9
    end

    @testset "c1: FDT pipeline smoke on the device path (loose)" begin
        H1 = MC.TiledHamiltonian(_uniaxial_model(0.05); dims = (1, 1, 1))
        gH1 = MC.GPUTiledHamiltonian(CPU(), H1)
        kt = 0.03
        prob = LLGProblem(H1; magmom = 2.0, alpha = 1.0)
        obs = [Observable(:ez2, 1, v -> v.config[1][3]^2)]
        r = SD.gpu_run_llg(prob, _uniaxial_config(0.3), gH1; dt = 0.05,
                           nsteps = 60_000, kT = kt, seed = 42,
                           measure_interval = 10, observables = obs)
        st = equilibrium_stats(r; evaluables = Evaluable[])
        Eu(u) = total_energy(H1, _uniaxial_config(u))
        ez2 = _boltzmann_average(Eu, kt, (u, E) -> u^2)
        @test abs(st[:ez2].mean[1] - ez2) < max(4 * st[:ez2].err[1], 0.1 * ez2)
    end

    @testset "validation" begin
        @test_throws ArgumentError SD.gpu_run_llg(probd, c0d, gHd; dt = 0.05,
                                                  nsteps = 1, workgroupsize = 3)
        @test_throws ArgumentError SD.gpu_run_llg(probd, c0d, gHd; dt = 0.05,
                                                  nsteps = 1, seed = 1)
        H2 = MC.TiledHamiltonian(_biquadratic_model(0); dims = (3, 1, 1))
        gH2 = MC.GPUTiledHamiltonian(CPU(), H2)
        @test_throws ArgumentError SD.gpu_run_llg(probd, c0d, gH2; dt = 0.05,
                                                  nsteps = 1)
        # the same config0 entry door as run_llg (refuse off-band, project
        # the near-pole 5e-9 case that used to DomainError mid-run)
        bad = copy(c0d)
        bad[1] = SVector(2.0, 0.0, 0.0)
        @test_throws ArgumentError SD.gpu_run_llg(probd, bad, gHd; dt = 0.05,
                                                  nsteps = 1)
        pole = copy(c0d)
        pole[1] = SVector(0.0, 0.0, 1.0 + 5.0e-9)
        rp = SD.gpu_run_llg(probd, pole, gHd; dt = 0.05, nsteps = 2)
        @test all(isfinite(c) for e in rp.config for c in e)
    end
end
