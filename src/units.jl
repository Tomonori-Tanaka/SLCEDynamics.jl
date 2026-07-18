# Physical constants. `HBAR_EV_FS` is an exact ratio of SI defining constants
# (h and e are exact since the 2019 SI); `MU_B_EV_T` carries the CODATA-2018
# measured μ_B. `KB_EV` (for the stochastic thermostat) is NOT redefined here —
# reuse `SCEMonteCarlo.KB_EV` / `SCEMonteCarlo.resolve_kt`.

"""
    HBAR_EV_FS

The reduced Planck constant ħ in eV·fs, `h/(2π·e)·10¹⁵` from the exact SI values
of `h` and `e`: `0.6582119569509066…`. The internal unit system of this package:
energies in the model's units (eV), time in fs, so a gradient magnitude `|G|` [eV]
on a moment `magmom` [μ_B] precesses at `g·|G|/(ħ·magmom)` [rad/fs].
"""
const HBAR_EV_FS = 6.62607015e-34 / (2π * 1.602176634e-19) * 1e15

"""
    MU_B_EV_T

The Bohr magneton μ_B in eV/T (CODATA-2018: `5.7883818060e-5`). Used only at the
tesla boundary — the Zeeman energy `E_Z = −Σ_i magmom_i·μ_B·(e_i · B_ext)` of an
external field in tesla; the core LLG evolution never touches it (μ_B cancels).
A `g = 2` spin in `B = 1 T` precesses at `2·MU_B_EV_T/(2π·HBAR_EV_FS)` = 27.9925
GHz — the unit sanity anchor pinned in `test_units.jl`.
"""
const MU_B_EV_T = 9.2740100783e-24 / 1.602176634e-19
