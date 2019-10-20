import numpy as np
from pyncband import CoreShellParticle2, Material2
from pyncband.constants import ev_to_hartree as ev2hartree
# from pyncband.physicsfunctions2 import

from pyncband.physicsfunctions2 import e2k, k2e, _tanxdivx

def test_wavefunction_and_derivative_continuity():
    a = Material2(1.0, -2, 1.0, 1.0, 1.0)
    b = Material2(1.0, -1.5, 1.0, 1.0, 1.0)

    cs = CoreShellParticle2(a, b)
    print(cs._calculate_s1_energies(1.0, 1.0))

test_wavefunction_and_derivative_continuity()