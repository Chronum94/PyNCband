from scipy.constants import hbar, e

# hbar in eV
hbar_ev = hbar / e

# A nano- conversion factor. This may be deprecated/removed.
n_ = 1e-9

# This converts the calculated wavenumber when we use masses of kg, energies in eV, and hbar in eV-s.
# The wavenumber comes out in units of <this number> / nm.
wavenumber_nm_from_energy_ev = 2.498301248024997
