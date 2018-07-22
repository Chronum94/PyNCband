import matplotlib.pyplot as plt
import numpy as np

from pyncband import Material, CoreShellParticle

mat1 = Material(1.0, -0.5, 0.1, 0.6, 4.0)
mat2 = Material(1.0, 0.0, 0.1, 0.6, 4.0)

ehqd = CoreShellParticle(mat1, mat2, 1.0, 1.0)
heqd = CoreShellParticle(mat2, mat1, 1.0, 1.0)
print(ehqd.numerical_overlap_integral())
print(heqd.numerical_overlap_integral())
shell_radii = np.linspace(0.2, 2, 10)
ae = np.vectorize(ehqd.localization_electron_core)
ah = np.vectorize(ehqd.localization_electron_core)
plt.plot(shell_radii, ae(shell_radii), shell_radii, ah(shell_radii))
plt.show()

