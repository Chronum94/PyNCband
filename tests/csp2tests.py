import numpy as np
from pyncband import CoreShellParticle2, Material2
from pyncband.constants import ev_to_hartree as ev2hartree
# from pyncband.physicsfunctions2 import

from pyncband.physicsfunctions2 import e2k, k2e, _tanxdivx

def test_wavefunction_and_derivative_continuity():
    a = Material2(1.0, -2, 0.9, 1.0, 1.0)
    b = Material2(1.0, -1.9, 0.09, 1.0, 1.0)

    core_radius, shell_thickness = 10, 10
    cs = CoreShellParticle2(a, b)
    es1, hs1 = cs._calculate_s1_energies(core_radius, shell_thickness)
    cs._plot_wf(es1, hs1, core_radius, shell_thickness)
    print()

test_wavefunction_and_derivative_continuity()