import numpy as np
import matplotlib.pyplot as plt

from pyncband import CoreShellParticle2, Material2
from pyncband.constants import ev_to_hartree as ev2hartree

# from pyncband.physicsfunctions2 import

from pyncband.physicsfunctions2 import e2k, k2e, _tanxdivx


def test_wavefunction_and_derivative_continuity():
    a = Material2(1.0, -2, 0.018, 0.05, 1.0)
    b = Material2(1.1, -1.5, 0.09, 0.11, 1.0)

    core_radius, shell_thickness = 60, 60
    cs = CoreShellParticle2(a, b)
    es1, hs1 = cs._calculate_s1_energies(core_radius, shell_thickness)
    cs._plot_wf(es1, hs1, core_radius, shell_thickness)
    print()


def test_s1_energies():
    a = Material2(1.0, -2, 0.018, 0.05, 1.0)
    b = Material2(1.1, -1.5, 0.09, 0.11, 1.0)

    core_radius, shell_thickness = 60, 60
    cs = CoreShellParticle2(a, b)

    npts = 100
    r = np.linspace(20, 300, npts)
    corex, shellx = np.meshgrid(r, r)
    s1_energies = np.zeros_like(corex)

    for i in range(npts):
        for j in range(npts):
            s1_energies[i, j] = np.sum(cs._calculate_s1_energies(corex[i, j], shellx[i,j]))

    plt.contourf(corex / 18.8973, shellx / 18.8973, s1_energies, levels = np.linspace(0, 0.25, 50), extend='both')
    plt.colorbar()
    plt.show()


# test_wavefunction_and_derivative_continuity()
test_s1_energies()
