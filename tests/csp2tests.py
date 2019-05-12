import numpy as np

from pyncband import CoreShellParticle2, Material

a, b = Material(1, -1, 1, 1, 1), Material(0.8, -0.6, 1, 1, 1)

cs = CoreShellParticle2(a, b)
assert np.isclose(cs.core_electron_potential_offset, 0)
assert np.isclose(cs.shell_electron_potential_offset, 0.4)
assert np.isclose(cs.core_hole_potential_offset, 0.6)
assert np.isclose(cs.shell_hole_potential_offset, 0)