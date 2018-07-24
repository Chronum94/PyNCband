"""CSNC = CoreShell semi-conductor nanocrystal.

"""

import numpy as np
from numpy import testing as test

from pyncband import Material, CoreShellParticle


def test_type_two_and_he():
    """Tests for CSNC type-2 and h/e structure.

    """
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc = CoreShellParticle(a, b, 1.0, 1.0)
    assert csnc.type_two  # This is a type 2 CSNC.
    assert csnc.h_e  # This is an h/e structure.


def test_type_two_and_eh():
    """Tests for CSNC type-2 and e/h structure.

    """
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc = CoreShellParticle(b, a, 1.0, 1.0)
    assert csnc.type_two  # This is a type 2 CSNC.
    assert csnc.e_h  # This is an h/e structure.


def test_type_one():
    """Tests for CSNC type-1.

    """
    a = Material(2.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc = CoreShellParticle(b, a, 1.0, 1.0)
    assert csnc.type_one  # This is a type 1 CSNC.


def test_equal_wavenumbers_for_type_one():
    """Tests if the wavenumbers of the electron/hole wavefunctions are equal, given all relevant variables are equal.

    """
    a = Material(2.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc = CoreShellParticle(b, a, 1.0, 1.0)  # Type 1 CSNC.
    electron_core_wavenumber, electron_shell_wavenumber, hole_core_wavenumber, hole_shell_wavenumber = (
        csnc.calculate_wavenumbers()
    )
    test.assert_allclose(electron_core_wavenumber, hole_core_wavenumber)
    test.assert_allclose(electron_shell_wavenumber, hole_shell_wavenumber)


def test_equal_wavenumbers_for_type_two_eh():
    """Tests if the wavenumbers of the electron/hole wavefunctions are equal, given all relevant variables are equal.

    """
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc = CoreShellParticle(a, b, 1.0, 1.0)  # Type 1 CSNC.
    electron_core_wavenumber, electron_shell_wavenumber, hole_core_wavenumber, hole_shell_wavenumber = (
        csnc.calculate_wavenumbers()
    )
    test.assert_allclose(electron_core_wavenumber, hole_shell_wavenumber)
    test.assert_allclose(electron_shell_wavenumber, hole_core_wavenumber)


def test_tighter_confinement_leads_to_higher_energy_states():
    """Tests that a smaller CSNC leads to higher-energy eigenstates.

    """
    a = Material(2.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(b, a, 1.0, 1.0)  # Type 1 CSNC.
    csnc2 = CoreShellParticle(b, a, 0.9, 1.0)  # Type 1 CSNC.
    csnc1_energies = csnc1.calculate_s1_energies()
    csnc2_energies = csnc2.calculate_s1_energies()
    assert np.all(csnc2_energies > csnc1_energies)


def test_thicker_shell_requires_smaller_core_for_localization_eh():
    """Tests that a thicker shell leads to carrier confinement in a smaller core, in type-2 e/h CSNCs.

    """

    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(b, a, 1.0, 1.0)  # Type 2 CSNC.
    csnc2 = CoreShellParticle(b, a, 1.0, 1.1)  # Type 2 CSNC.
    csnc1_coreloc = csnc1.localization_electron_core()
    csnc2_coreloc = csnc2.localization_electron_core()
    assert csnc2_coreloc < csnc1_coreloc


def test_thicker_shell_requires_smaller_core_for_localization_he():
    """Tests that a thicker shell leads to carrier confinement in a smaller core, in type-2 h/e CSNCs.

    """
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC, h/e structure.
    csnc2 = CoreShellParticle(a, b, 1.0, 1.1)  # Type 2 CSNC, h/e structure.
    csnc1_coreloc = csnc1.localization_hole_core()
    csnc2_coreloc = csnc2.localization_hole_core()
    assert csnc2_coreloc < csnc1_coreloc


def test_thicker_core_requires_smaller_shell_for_localization_he():
    """Tests that a thicker core leads to carrier confinement in a thinner shell, in type-2 h/e CSNCs.

    """
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC. h/e structure.
    csnc2 = CoreShellParticle(a, b, 1.1, 1.0)  # Type 2 CSNC. h/e structure.
    csnc1_shellloc = csnc1.localization_electron_shell()
    csnc2_shellloc = csnc2.localization_electron_shell()
    assert csnc2_shellloc < csnc1_shellloc


def test_thicker_core_requires_smaller_shell_for_localization_eh():
    """Tests that a thicker core leads to carrier confinement in a thinner shell, in type-2 e/h CSNCs.

    """
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(b, a, 1.0, 1.0)  # Type 2 CSNC. e/h structure.
    csnc2 = CoreShellParticle(b, a, 1.1, 1.0)  # Type 2 CSNC. e/h structure.
    csnc1_shellloc = csnc1.localization_hole_shell()
    csnc2_shellloc = csnc2.localization_hole_shell()
    assert csnc2_shellloc < csnc1_shellloc


def test_adaptive_energy_bracketing_for_high_energies():
    """Tests that high energies are bracketed for extremely small CSNCs.

    """
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 0.1, 0.1)  # Type 2 CSNC. h/e structure.
    energies = csnc1.calculate_s1_energies()
    # print(energies)
    assert np.isclose(energies[0], energies[1])


def test_wavenumbers_are_order_unity():
    """Tests that wavenumbers are approximately around 1.

    """
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC. h/e structure.
    wavenumbers = csnc1.calculate_wavenumbers()
    assert np.all(np.logical_and(1e-1 < abs(wavenumbers), abs(wavenumbers) < 10))


def test_energies_are_order_unity():
    """Tests that energies are approximately around 1.

    """
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC. h/e structure.
    energies = csnc1.calculate_s1_energies()
    assert np.all(np.logical_and(1e-1 < energies, energies < 10))


def test_numerical_overlap_integral_is_at_or_below_one():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC. h/e structure.
    assert csnc1.numerical_overlap_integral() <= 1.0


def test_analytical_overlap_integral_is_at_or_below_one():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC. h/e structure.
    assert csnc1.analytical_overlap_integral() <= 1.0


def test_overlap_integrals_are_consistent():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC. h/e structure.
    assert np.isclose(csnc1.analytical_overlap_integral(), csnc1.numerical_overlap_integral())


def test_asymptotic_core_localization():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC. h/e structure.
    csnc2 = CoreShellParticle(b, a, 1.0, 1.0)  # Type 2 CSNC. e/h structure.
    assert np.isclose(csnc1.localization_hole_core(50000), csnc1. localization_hole_core(asymp=True))
    assert np.isclose(csnc2.localization_electron_core(50000), csnc2.localization_electron_core(asymp=True))

def test_asymptotic_shell_localization():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC. h/e structure.
    csnc2 = CoreShellParticle(b, a, 1.0, 1.0)  # Type 2 CSNC. e/h structure.
    assert np.isclose(csnc1.localization_electron_shell(1e6), csnc1. localization_electron_shell(asymp=True))
    assert np.isclose(csnc2.localization_hole_shell(1e6), csnc2.localization_hole_shell(asymp=True))

test_asymptotic_core_localization()