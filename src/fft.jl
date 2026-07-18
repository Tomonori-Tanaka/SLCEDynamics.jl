# A minimal power-of-two complex FFT for the S(q,ω) estimator. Deliberately NOT
# FFTW: the transform cost is negligible next to the spatial phase sum (SPEC), and
# an own fixed-order kernel keeps `structure_factor` a pure, bit-reproducible
# function of its inputs for any task count (FFTW results can vary with wisdom /
# planner state / threading). Gated against an O(n²) reference DFT in
# `test_sqw_core.jl` — change either side and re-run it.

# Precomputed twiddle table for length-`n` transforms (n a power of two):
# w[j+1] = exp(−2πi·j/n), j = 0 … n/2−1 — the forward-DFT kernel sign.
struct _Twiddle
    n::Int
    w::Vector{ComplexF64}
    function _Twiddle(n::Int)
        (n >= 2 && ispow2(n)) ||
            throw(ArgumentError("FFT length must be a power of two ≥ 2; got $n"))
        return new(n, [cis(-2π * j / n) for j = 0:(n >>> 1)-1])
    end
end

# In-place bit-reversal permutation (the standard iterative form).
function _bitrev!(a::AbstractVector{ComplexF64})::Nothing
    n = length(a)
    j = 1
    @inbounds for i = 1:n-1
        if i < j
            a[i], a[j] = a[j], a[i]
        end
        k = n >>> 1
        while k < j
            j -= k
            k >>>= 1
        end
        j += k
    end
    return nothing
end

# In-place iterative radix-2 decimation-in-time forward FFT:
# a[k+1] ← Σ_n a[n+1]·exp(−2πi·k·n/N). Fixed summation order — deterministic.
function _fft_pow2!(a::AbstractVector{ComplexF64}, tw::_Twiddle)::Nothing
    n = tw.n
    length(a) == n || throw(DimensionMismatch(
        "FFT buffer has length $(length(a)); twiddle table is for $n"))
    _bitrev!(a)
    len = 2
    @inbounds while len <= n
        half = len >>> 1
        step = div(n, len)
        for base = 0:len:n-1
            for j = 0:half-1
                wj = tw.w[j*step+1]
                u = a[base+j+1]
                v = a[base+j+half+1] * wj
                a[base+j+1] = u + v
                a[base+j+half+1] = u - v
            end
        end
        len <<= 1
    end
    return nothing
end

# The window samples w_n (n = 0 … M−1). `:hann` is the periodic form (the DFT
# convention); `:none` is rectangular.
function _fill_window!(w::Vector{Float64}, window::Symbol)::Nothing
    M = length(w)
    if window === :hann
        for n = 0:M-1
            w[n+1] = 0.5 * (1 - cos(2π * n / M))
        end
    elseif window === :none
        fill!(w, 1.0)
    else
        throw(ArgumentError("window must be :hann or :none; got :$window"))
    end
    return nothing
end

# W₂ = (1/M)Σ w_n² — the window power that normalizes the spectral density.
_window_power(w::Vector{Float64})::Float64 = sum(abs2, w) / length(w)

# The two-sided fftshifted frequency axis: ω_k′ = 2πk′/(M·Δt) for
# k′ = −M/2 … M/2−1 (ascending; the Nyquist bin appears only at −M/2).
_freq_axis(M::Int, dt::Float64)::Vector{Float64} =
    [2π * k / (M * dt) for k = -(M >>> 1):(M>>>1)-1]

# DFT index (1-based) of the signed bin k′ ∈ [−M/2, M/2−1].
_dft_index(kp::Int, M::Int)::Int = mod(kp, M) + 1
