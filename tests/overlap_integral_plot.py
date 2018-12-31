import matplotlib.pyplot as plt
import numpy as np

from pyncband import Material, CoreShellParticle

inp_effective_electron_mass = 0.1
InP = Material(1.34, 0, inp_effective_electron_mass, 0.64, 9.6, "InP")
CdS = Material(2.20, -0.39, 0.21, 0.68, 5.3, "CdS")
csnc = CoreShellParticle(InP, CdS, 1.0, 1.0, 1.0)


def overlap_integral(core_width, shell_width):
    csnc.set_shell_width(shell_width)
    csnc.set_core_width(core_width)
    # bg = csnc.bandgap
    # state_nrg = csnc.calculate_s1_energies()
    # col_nrg = csnc.coulomb_screening_energy()
    # pol_nrg = csnc.polarization_screening_energy()

    return (
        csnc.numerical_overlap_integral()
    )  # bg + np.sum(state_nrg) + col_nrg[0] + pol_nrg[0]


voverlap_integral = np.vectorize(overlap_integral)
x = np.linspace(0.1, 5.0, 75)
xx, yy = np.meshgrid(x, x)
zz = voverlap_integral(xx, yy)

levels = np.linspace(np.min(zz), np.max(zz), 20)
plt.contourf(zz, cmap="jet", levels=levels)

plt.colorbar()
plt.show()
# print(overlap_integral(0.5, 0.5))

plt.plot(x, voverlap_integral(2, x), x, voverlap_integral(x, 2))
plt.show()
#
# InP = Material(1.34, 0, 0.1, 0.64, 9.6, "InP")
# CdS = Material(2.20, -0.39, 0.21, 0.68, 5.3, "CdS")
# csnc = CoreShellParticle(InP, CdS, 0.5, 0.5, 1.0)
# y = csnc.plot_electron_wavefunction()
# print(csnc.analytical_overlap_integral(), csnc.numerical_overlap_integral())
# # plt.plot(y.real)
# # plt.plot(y.imag)
# plt.plot(np.gradient(y.real))
# plt.show()
