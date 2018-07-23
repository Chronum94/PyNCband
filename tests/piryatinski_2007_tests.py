import matplotlib.pyplot as plt
import numpy as np

from pyncband import Material, CoreShellParticle

mat1 = Material(1.0, -0.5, 0.1, 0.6, 4.0)
mat2 = Material(1.0, 0.0, 0.1, 0.6, 4.0)

ehqd = CoreShellParticle(mat1, mat2, 1.0, 1.0)
heqd = CoreShellParticle(mat2, mat1, 1.0, 1.0)
print(ehqd.numerical_overlap_integral())
print(heqd.numerical_overlap_integral())
shell_radii = np.linspace(0.2, 2, 50)
core_radii = np.linspace(0.5, 4, 50)


def overlap(corew, shellw):
    ehqd.set_core_width(corew)
    ehqd.set_shell_width(shellw)
    return ehqd.analytical_overlap_integral()


ae = np.vectorize(ehqd.localization_electron_core)
ah = np.vectorize(ehqd.localization_electron_core)
voverlap = np.vectorize(overlap)

ss, rr = np.meshgrid(shell_radii, core_radii)
overlap_integral = voverlap(ss, rr)

clevels = np.linspace(0, 1, 11)
a = plt.contourf(overlap_integral, levels=clevels)
plt.clabel(a)
plt.colorbar()
plt.show()

# plt.plot(shell_radii, ae(shell_radii), shell_radii, ah(shell_radii))
