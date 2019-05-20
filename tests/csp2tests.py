import numpy as np
from scipy.constants import Rydberg
from pyncband import CoreShellParticle, CoreShellParticle2, Material
from pyncband.constants import ev_to_hartree, hartree_to_ev


def test_base_coreshellparticle2():
    a, b = Material(1, -1, 1, 1, 1), Material(0.8, -0.6, 1, 1, 1)
    cs2 = CoreShellParticle2(a, b)
    cs1 = CoreShellParticle(a, b, 1.0, 1.0)
    assert np.isclose(cs2.core_electron_potential_offset, 0)
    assert np.isclose(cs2.shell_electron_potential_offset, 0.4 * ev_to_hartree)
    assert np.isclose(cs2.core_hole_potential_offset, 0.6 * ev_to_hartree)
    assert np.isclose(cs2.shell_hole_potential_offset, 0)

def test_cs1_cs2_compat():
    a, b = Material(1, -1, 1, 1, 1), Material(0.8, -0.6, 1, 1, 1)
    cs2 = CoreShellParticle2(a, b)
    cs1 = CoreShellParticle(a, b, 1.0, 1.0)

    assert np.isclose(cs2.shell_electron_potential_offset, 0.4 * ev_to_hartree, cs1.ue)
    assert np.isclose(cs2.core_hole_potential_offset, 0.4 * ev_to_hartree, cs1.uh)

    assert np


def test_s1_energies():
    a, b = Material(1, -1, 1, 1, 1), Material(0.8, -0.6, 1, 1, 1)
    cs = CoreShellParticle2(a, b)


def test_csp_csp2_answers_equal():
    """Tests that a old and new CSNCs offer equal answers.

    """
    a = Material(2.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.

    core, shell = 1.0, 1.0
    csnc1 = CoreShellParticle(b, a, core, shell)  # Type 1 CSNC.
    csnc2 = CoreShellParticle2(b, a)  # Type 1 CSNC.
    csnc1_energies = csnc1.calculate_s1_energies()
    csnc2_energies = csnc2._calculate_s1_energies(
        core / 0.52917721067, shell / 0.52917721067
    )  #  * hartree_to_ev
    print(csnc1_energies, csnc2_energies)
    # assert np.allclose(csnc1_energies, csnc2_energies)


test_csp_csp2_answers_equal()
