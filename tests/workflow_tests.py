import numpy as np
import numpy.testing as test
import matplotlib.pyplot as plt

from scipy.constants import e

from PyNCband.Material import Material
from PyNCband.CoreShellParticle import CoreShellParticle


n_ = 1e-9

AlSb = Material(1.6, -2.9, 0.12, 0.98, 2, "AlSb")
CdS = Material(2.4, -3.5, 0.21, 0.8, 3, "CdS")
test.assert_almost_equal(AlSb.vbe, -4.5)
test.assert_almost_equal(CdS.vbe, -5.9)

CdS_AlSb = CoreShellParticle(CdS, AlSb, 0.5, 0.3)
AlSb_CdS = CoreShellParticle(AlSb, CdS, 0.5, 0.3)
assert CdS_AlSb.type_two, "CdS/AlSb is is a type 2 QD"
assert CdS_AlSb.e_h, "CdS/AlSb should be e-h core-shell."

x = np.linspace(1, 2, 10)

xx, yy = np.meshgrid(x, x)
zz = np.zeros_like(xx)

# @profile

# for i, cw in enumerate(x):
#     for j, sw in enumerate(x):
#         CdS_AlSb.set_core_width(cw)
#         CdS_AlSb.set_shell_width(sw)
#         zz[i, j] = CdS_AlSb.analytical_overlap_integral()
#         test.assert_approx_equal(zz[i, j], CdS_AlSb.numerical_overlap_integral())

CdS_AlSb.set_core_width(1)
CdS_AlSb.set_shell_width(1)
print(CdS_AlSb.ue / e)
for shellw in np.linspace(0.1, 40, 10):
    print("Shell width:", shellw)
    print(CdS_AlSb.localization_electron_min_width(shellw) / n_, '\n')

# plt.imshow(zz)
# plt.colorbar()
# # plt.clim(0, 20)
# plt.show()
