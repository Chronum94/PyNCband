import warnings as w
from scipy.integrate import IntegrationWarning
import numpy as np
import numpy.testing as test
import matplotlib.pyplot as plt

from scipy.constants import e

from PyNCband.Material import Material
from PyNCband.CoreShellParticle import CoreShellParticle
from PyNCband.scaling import n_

AlSb = Material(1.6, -2.9, 0.12, 0.98, 11.0, "AlSb")
CdS = Material(2.4, -3.5, 0.21, 0.8, 9.4, "CdS")
test.assert_almost_equal(AlSb.vbe, -4.5)
test.assert_almost_equal(CdS.vbe, -5.9)

CdS_AlSb = CoreShellParticle(CdS, AlSb, 0.5, 0.3)
AlSb_CdS = CoreShellParticle(AlSb, CdS, 0.5, 0.3)
assert CdS_AlSb.type_two, "CdS/AlSb is is a type 2 QD"
assert CdS_AlSb.e_h, "CdS/AlSb should be e-h core-shell."

x = np.linspace(1, 2, 10)

xx, yy = np.meshgrid(x, x)
zz = np.zeros_like(xx)


CdS_AlSb.set_core_width(0.7)
# CdS_AlSb.set_shell_width(4)
with w.catch_warnings() as wfil:
    w.simplefilter("error", RuntimeWarning)
    print("CBE step in eV:", CdS_AlSb.ue / e)
    for shellw in np.linspace(0.501, 0.505, 4):
        # print("Width:", shellw)
        print("Coreloc:", CdS_AlSb.localization_electron_core(shellw))
        print("Shellloc:", CdS_AlSb.localization_hole_shell(shellw))
        CdS_AlSb.set_shell_width(shellw)
        energies = np.array(CdS_AlSb.calculate_s1_energies()) / e
        coulomb_energy = CdS_AlSb.coulomb_screening_energy()
        polarization_energy = CdS_AlSb.interface_polarization_energy()
        print("EG", energies)
        print("CE:", coulomb_energy)
        print("PE:", polarization_energy)
        print(
            CdS_AlSb.analytical_overlap_integral(),
            CdS_AlSb.numerical_overlap_integral(),
        )
        # print("BG:", CdS_AlSb.bandgap)
        # print("ExcEnergy:", CdS_AlSb.bandgap + np.sum(energies) + coulomb_energy[0] + polarization_energy[0])
        # print(AlSb_CdS.localization_electron_core())
        print("\n")

CdS_AlSb.plot_potential_profile()
