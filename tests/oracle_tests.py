import numpy.testing as test
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import hbar, m_e, e

from PyNCband.Material import Material
from PyNCband.CoreShellParticle import CoreShellParticle

n_ = 1e-9
# These tests have known answers.

# In the units branch, the masses are in units of electron mass, energies in eV, and epsilons in eps0.
# They are internally converted as need be.
# Bandgap, cbe, m_e, m_h, eps, name.
all_one_core = Material(1, -1, 1, 1, 1, "Ones")
all_one_shell = Material(1, -0.5, 1, 1, 1, "Ones")
eff_mass_test = Material(1, -1, 0.5, 1, 1, "ElecMassHalf")

# In the units branch, enter lengths in nm.
coreshell_1 = CoreShellParticle(all_one_core, all_one_shell, 0.5, 0.3)
coreshell_2 = CoreShellParticle(all_one_core, all_one_shell, 0.6, 0.6)
assert coreshell_1.e_h
assert coreshell_2.e_h
cs1_s1_energies = coreshell_1.calculate_s1_energies()
cs2_s1_energies = coreshell_2.calculate_s1_energies()
print("S1 electron and hole energies:   \n", cs1_s1_energies)
print("S1 electron and hole energies:\n", cs2_s1_energies)
test.assert_allclose(cs1_s1_energies, (0.7032032032032032, 0.7032032032032032))
test.assert_allclose(cs2_s1_energies, (0.4554554554554554, 0.4554554554554554))

cs1_s1_wavenumbers = coreshell_1.calculate_wavenumbers()
cs2_s1_wavenumbers = coreshell_2.calculate_wavenumbers()
print("Electron k in c/s, hole k in c/s:\n", cs1_s1_wavenumbers)
print("Electron k in c/s, hole k in c/s:\n", cs2_s1_wavenumbers)
test.assert_allclose(
    cs1_s1_wavenumbers,
    (4296145225.210343, 2309424710.175334, 2309424710.175334, 4296145225.210343),
)
test.assert_allclose(
    cs2_s1_wavenumbers,
    (3457494418.5655584, 1081274086.3061583j, 1081274086.3061583j, 3457494418.5655584),
)

print("Analytical overlap integral:", coreshell_1.analytical_overlap_integral())
print("Numerical overlap integral:", coreshell_1.numerical_overlap_integral())
print("Min core for e-loc:", coreshell_1.localization_electron_min_width() / n_)
print("Min shell for h-loc:", coreshell_1.localization_hole_min_width() / n_)
print(
    "Min core LIMIT for e-loc:", coreshell_1.localization_electron_min_width(1e5) / n_
)
print("Min shell LIMIT for h-loc:", coreshell_1.localization_hole_min_width(1e5) / n_)
print("Min core for e-loc:", coreshell_2.localization_electron_min_width() / n_)
print("Min shell for h-loc:", coreshell_2.localization_hole_min_width() / n_)
print(
    "Min core LIMIT for e-loc:", coreshell_2.localization_electron_min_width(1e5) / n_
)
print("Min shell LIMIT for h-loc:", coreshell_2.localization_hole_min_width(1e5) / n_)
print("Coulomb screening energy:", coreshell_1.coulomb_screening_energy())
print("Interface polarization energy:", coreshell_1.interface_polarization_energy())
print("Coulomb screening energy:", coreshell_2.coulomb_screening_energy())
print("Interface polarization energy:", coreshell_2.interface_polarization_energy())
# coreshell_1.plot_potential_profile()
# plt.plot(
#     np.abs(
#         coreshell_1.plot_electron_wavefunction(
#             np.linspace(0, 3, 100), 1.715268410618755, 1.393608883606369
#         )
#     )
# )
# plt.show()
#
# plt.plot(
#     np.abs(
#         coreshell_2.plot_electron_wavefunction(
#             np.linspace(0, 6, 100), 1.715268410618755, 1.393608883606369
#         )
#     )
# )
# plt.show()
