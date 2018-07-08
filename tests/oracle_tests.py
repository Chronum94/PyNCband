import numpy.testing as test
import numpy as np
import matplotlib.pyplot as plt

from PyNCband.Material import Material
from PyNCband.CoreShellParticle import CoreShellParticle

# These tests have known answers.

all_one_core = Material(1, - 1, 1, 1, 1, 'Ones')
all_one_shell = Material(1, - 0.5, 1, 1, 1, 'Ones')
eff_mass_test = Material(1, 1, 1, 0.5, 1, 'HoleMassHalf')

coreshell_1 = CoreShellParticle(all_one_core, all_one_shell, 1, 1)

assert coreshell_1.eh == True
print(coreshell_1.calculate_s1_energies())
test.assert_allclose(coreshell_1.calculate_s1_energies(), (1.4710728602332950, 1.4710728602332950))
print(coreshell_1.calculate_electron_wavevectors())
test.assert_allclose(coreshell_1.calculate_electron_wavevectors(), (1.715268410618755, 1.393608883606369))
# test.assert_allclose(c)