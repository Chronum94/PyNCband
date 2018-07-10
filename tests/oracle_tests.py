import numpy.testing as test
import numpy as np
import matplotlib.pyplot as plt

from PyNCband.Material import Material
from PyNCband.CoreShellParticle import CoreShellParticle

# These tests have known answers.

all_one_core = Material(1, -1, 1, 1, 1, "Ones")
all_one_shell = Material(1, -0.5, 1, 1, 1, "Ones")
eff_mass_test = Material(1, 1, 1, 0.5, 1, "HoleMassHalf")

coreshell_1 = CoreShellParticle(all_one_core, all_one_shell, 1., 1)

assert coreshell_1.e_h == True
print("S1 electron and hole energies:\n", coreshell_1.calculate_s1_energies())
s1_energies = coreshell_1.calculate_s1_energies()
test.assert_allclose(s1_energies, (1.4710728602332950, 1.4710728602332950))

s1_wavevectors = coreshell_1.calculate_wavevectors()
print("Electron k in c/s, hole k in c/s:\n", s1_wavevectors)
test.assert_allclose(
    s1_wavevectors,
    (1.715268410618755, 1.393608883606369, 1.393608883606369, 1.715268410618755),
)
print("Analytical overlap integral:", coreshell_1.analytical_overlap_integral())
print("Numerical overlap integral:", coreshell_1.numerical_overlap_integral())
print("Min core for e-loc:", coreshell_1.localization_electron_min_width())
print("Min shell for h-loc:", coreshell_1.localization_hole_min_radius())
print("Coulomb screening energy:", coreshell_1.coulomb_screening_energy())
print("Interface polarization energy:", coreshell_1.interface_polarization_energy())
# coreshell_1.plot_potential_profile()
plt.plot(
    np.abs(
        coreshell_1.plot_electron_wavefunction(
            np.linspace(0, 3, 100), 1.715268410618755, 1.393608883606369
        )
    )
)
plt.show()
