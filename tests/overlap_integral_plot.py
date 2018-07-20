import numpy as np
import matplotlib.pyplot as plt

from pyncband import Material, CoreShellParticle


def overlap_integral(core_width, shell_width):
    inp_effective_electron_mass = 0.1
    InP = Material(1.34, 0, inp_effective_electron_mass, 0.64, 9.6, "InP")
    CdS = Material(2.20, -0.39, 0.21, 0.68, 5.3, "CdS")
    csnc = CoreShellParticle(InP, CdS, core_width, shell_width, 1.0)
    return csnc.numerical_overlap_integral()  # , csnc.numerical_overlap_integral()


# voverlap_integral = np.vectorize(overlap_integral)
# x = np.linspace(0.5, 10, 25)
# xx, yy = np.meshgrid(x, x)
# zz = voverlap_integral(xx, yy)
#
# # plt.imshow(zz, extent=[0.5, 10, 10, 0.5])
# # plt.colorbar()
# # plt.show()
# # print(overlap_integral(0.5, 0.5))
#
# plt.plot(x, voverlap_integral(6, x))
#
# plt.show()

InP = Material(1.34, 0, 0.1, 0.64, 9.6, "InP")
CdS = Material(2.20, -0.39, 0.21, 0.68, 5.3, "CdS")
csnc = CoreShellParticle(InP, CdS, 0.5, 0.5, 1.0)
y = csnc.plot_electron_wavefunction()
print(csnc.analytical_overlap_integral(), csnc.numerical_overlap_integral())
# plt.plot(y.real)
# plt.plot(y.imag)
plt.plot(np.gradient(y.real))
plt.show()
