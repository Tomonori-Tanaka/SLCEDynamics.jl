# Constants and the unit-system sanity anchor.

@testset "units" begin
    @testset "constants" begin
        # ħ in eV·fs: exact ratio of SI defining constants
        @test SD.HBAR_EV_FS ≈ 0.6582119569509066 rtol = 1e-15
        @test SD.HBAR_EV_FS == 6.62607015e-34 / (2π * 1.602176634e-19) * 1e15
        # μ_B in eV/T (CODATA-2018)
        @test SD.MU_B_EV_T ≈ 5.7883818060e-5 rtol = 1e-9
    end

    @testset "Larmor anchor: g = 2, B = 1 T → 27.9925 GHz" begin
        f_ghz = _larmor_omega(2.0, 1.0) / (2π) * 1e6      # 1/fs = 10⁶ GHz
        @test f_ghz ≈ 27.9925 rtol = 1e-5
        # period 35.724 ps
        @test 2π / _larmor_omega(2.0, 1.0) ≈ 35724.0 rtol = 1e-4
    end
end
