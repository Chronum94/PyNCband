import numpy as np
from pyncband import CoreShellParticle, CoreShellParticle2, Material
from pyncband.constants import ev_to_hartree, hartree_to_ev

from pyncband.physicsfunctions2 import e2k, k2e, _tanxdivx

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

def test_energy_wavevector_conversion():
    assert np.isclose(e2k(1, 1, 1), 0.0)
    assert np.isclose(e2k(3, 1, 1), 2.0)
    assert np.isclose(k2e(1, 1, 1), 1.5)
    assert np.isclose(e2k(1, 1, 3), 2.0j)
    assert np.isclose(k2e(2j, 1, 3), 5)

def test_s1_energies():
    a, b = Material(1, -1, 1, 1, 1), Material(0.8, -0.6, 1, 1, 1)
    cs = CoreShellParticle2(a, b)


def test_csp_csp2_answers_equal():
    """Tests that a old and new CSNCs offer equal answers.

    """
    a = Material(2.0, 0.0, 1.00, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    ang2bohr = 1.0 / 0.52917721067
    core, shell = 10.0, 10.0
    csnc1 = CoreShellParticle(b, a, core, shell)  # Type 1 CSNC.
    csnc2 = CoreShellParticle2(b, a)  # Type 1 CSNC.
    csnc1_energies = csnc1.calculate_s1_energies()
    csnc2_energies = csnc2._calculate_s1_energies(
        core * ang2bohr , shell * ang2bohr
    )  #  * hartree_to_ev
    print(csnc1_energies, csnc2_energies)
    print(_tanxdivx(0.01j), _tanxdivx(e2k(0.3, 1.0, 0.6))) # Verified.
    print()
    # assert np.allclose(csnc1_energies, csnc2_energies)


test_csp_csp2_answers_equal()
