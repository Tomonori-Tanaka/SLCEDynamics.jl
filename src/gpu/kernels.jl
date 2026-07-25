# The device step kernels: literal ports of `_omega` / `_rotate` / the two
# integrators' stage structure and of `_fill_noise!` / `_fill_noise_quantum!` /
# `_renormalize_active!` — one thread per site, same expression order as the
# host (the composite keyed reference in test_gpu_llg.jl and the bitwise gates
# rest on that). COUPLED SITES: change `_omega`/`_rotate`/`_step!`/
# `_fill_noise!`/`_qt_cascade!` on the host and these kernels (plus the gates)
# move with them.

# The thermal field, in-kernel: the SAME philox draws as the host `_fill_noise!`
# (same counter layout, same 2-blocks → 3-normals mapping, same fourth-normal
# discard) — bit-identical to the host draws on the KA-CPU backend (same libm),
# the same *realization* up to Box–Muller ULPs on CUDA. Inactive sites take the
# `dactive` branch and write an EXACT `zero(SVector)` — computing `σ·ξ` with
# `σ = 0` would produce −0.0 components for negative normals and break the
# bitwise gate against the host buffer (D12 in the decision record).
@kernel function _noise_kernel!(gth, @Const(sigma), @Const(active), seed::UInt64,
                                step::Int)
    s = @index(Global, Linear)
    @inbounds if active[s] != Int8(0)
        c1, c2 = _noise_ctrs(Int(s), step)
        n1, n2 = philox_normal2(philox_block(seed, c1))
        n3, _ = philox_normal2(philox_block(seed, c2))
        gth[s] = sigma[s] * SVector(n1, n2, n3)
    else
        gth[s] = zero(SVector{3,Float64})
    end
end

# One component's DF2T cascade update at site `s`, on the TRANSPOSED device
# state (`x[s, lane]` — site-fastest, coalesced): expression order identical
# to the host `_qt_cascade!` (COUPLED SITE — bitwise gate on the KA-CPU
# backend), only the indexing axis differs.
@inline function _qt_cascade_dev!(x, sections, s::Int, off::Int,
                                  xi::Float64)::Float64
    u = xi
    @inbounds for j in eachindex(sections)
        bq = sections[j]
        p1 = off + 2 * j - 1
        p2 = off + 2 * j
        out = bq.b0 * u + x[s, p1]
        x[s, p1] = bq.b1 * u - bq.a1 * out + x[s, p2]
        x[s, p2] = bq.b2 * u - bq.a2 * out
        u = out
    end
    return u
end

# The quantum-thermostat thermal field, in-kernel: the SAME slots-0/1 white
# draws as `_noise_kernel!` (shared realization), pushed through the biquad
# cascade — the literal port of the host `_fill_noise_quantum!`. Inactive
# sites take the `dactive` branch: exact `zero(SVector)` field, state columns
# untouched (the D12 discipline extended to the state array).
@kernel function _noise_kernel_quantum!(gth, xstate, @Const(sections),
                                        @Const(sigma), @Const(active),
                                        seed::UInt64, step::Int)
    s = @index(Global, Linear)
    # Int(s): the CUDA backend's @index returns Int32; the helpers are Int-typed
    @inbounds if active[s] != Int8(0)
        c1, c2 = _noise_ctrs(Int(s), step)
        n1, n2 = philox_normal2(philox_block(seed, c1))
        n3, _ = philox_normal2(philox_block(seed, c2))
        m = 2 * length(sections)
        y1 = _qt_cascade_dev!(xstate, sections, Int(s), 0, n1)
        y2 = _qt_cascade_dev!(xstate, sections, Int(s), m, n2)
        y3 = _qt_cascade_dev!(xstate, sections, Int(s), 2 * m, n3)
        gth[s] = sigma[s] * SVector(y1, y2, y3)
    else
        gth[s] = zero(SVector{3,Float64})
    end
end

# Depondt–Mertens stage 1: ω1 from (config, G, gth), predictor rotation.
@kernel function _dm_stage1_kernel!(epred, omega1, @Const(config), @Const(G),
                                    @Const(gth), @Const(pref), @Const(alpha),
                                    @Const(gzee), @Const(active), dt::Float64)
    s = @index(Global, Linear)
    @inbounds begin
        e = config[s]
        if active[s] == Int8(0)
            epred[s] = e
        else
            gt = G[s] + gzee[s] + gth[s]
            ω1 = -pref[s] * (gt + alpha[s] * cross(e, gt))
            omega1[s] = ω1
            epred[s] = _rotate(e, ω1 * dt)
        end
    end
end

# Depondt–Mertens stage 2: corrector rotation of the ORIGINAL spin.
@kernel function _dm_stage2_kernel!(config, @Const(epred), @Const(omega1),
                                    @Const(G), @Const(gth), @Const(pref),
                                    @Const(alpha), @Const(gzee), @Const(active),
                                    dt::Float64)
    s = @index(Global, Linear)
    @inbounds if active[s] != Int8(0)
        ep = epred[s]
        gt = G[s] + gzee[s] + gth[s]
        ω2 = -pref[s] * (gt + alpha[s] * cross(ep, gt))
        config[s] = _rotate(config[s], (omega1[s] + ω2) * (dt / 2))
    end
end

# Projected-Heun stage 1: Euler predictor + normalization.
@kernel function _hp_stage1_kernel!(epred, omega1, @Const(config), @Const(G),
                                    @Const(gth), @Const(pref), @Const(alpha),
                                    @Const(gzee), @Const(active), dt::Float64)
    s = @index(Global, Linear)
    @inbounds begin
        e = config[s]
        if active[s] == Int8(0)
            epred[s] = e
        else
            gt = G[s] + gzee[s] + gth[s]
            ω1 = -pref[s] * (gt + alpha[s] * cross(e, gt))
            omega1[s] = ω1
            ep = e + dt * cross(ω1, e)
            epred[s] = ep / norm(ep)
        end
    end
end

# Projected-Heun stage 2: trapezoidal corrector + normalization.
@kernel function _hp_stage2_kernel!(config, @Const(epred), @Const(omega1),
                                    @Const(G), @Const(gth), @Const(pref),
                                    @Const(alpha), @Const(gzee), @Const(active),
                                    dt::Float64)
    s = @index(Global, Linear)
    @inbounds if active[s] != Int8(0)
        ep = epred[s]
        gt = G[s] + gzee[s] + gth[s]
        ω2 = -pref[s] * (gt + alpha[s] * cross(ep, gt))
        e = config[s] + (dt / 2) * (cross(omega1[s], config[s]) + cross(ω2, ep))
        config[s] = e / norm(e)
    end
end

# Active-site renormalization (`e / norm(e)` is IEEE-exact arithmetic — bitwise
# equal to the host `_renormalize_active!`).
@kernel function _renorm_kernel!(config, @Const(active))
    s = @index(Global, Linear)
    @inbounds if active[s] != Int8(0)
        config[s] = config[s] / norm(config[s])
    end
end

# One queued device step (never synchronizes — KA queue order serializes the
# launches; the loop syncs only before a host download). The gradient calls
# always refresh the tesseral rows: both stages move every spin, so there is
# never a valid row reuse.
function _gpu_step!(integrator::AbstractIntegrator, st::GPULLGState, gH,
                    dt::Float64, ws::Int)::Nothing
    backend = gH.backend
    n = n_sites(gH.host)
    SLCEMonteCarlo.gpu_energy_gradient!(st.dG, gH, st.dconfig, st.gsc;
                                       workgroupsize = ws, synchronize = false)
    # invokelatest on every launch: the same static-analysis barrier as the
    # upstream GPU path (with an abstract-Backend signature the GPU half of the
    # kernel-invocation union has no method until a GPU package loads — a JET
    # false positive; one dynamic dispatch per launch is noise vs the kernel)
    k1 = integrator isa DepondtMertens ? _dm_stage1_kernel!(backend, ws) :
         _hp_stage1_kernel!(backend, ws)
    Base.invokelatest(k1, st.depred, st.domega1, st.dconfig, st.dG, st.dgth,
                      st.dpref, st.dalpha, st.dgzee, st.dactive, dt; ndrange = n)
    SLCEMonteCarlo.gpu_energy_gradient!(st.dG, gH, st.depred, st.gsc;
                                       workgroupsize = ws, synchronize = false)
    k2 = integrator isa DepondtMertens ? _dm_stage2_kernel!(backend, ws) :
         _hp_stage2_kernel!(backend, ws)
    Base.invokelatest(k2, st.dconfig, st.depred, st.domega1, st.dG, st.dgth,
                      st.dpref, st.dalpha, st.dgzee, st.dactive, dt; ndrange = n)
    return nothing
end
