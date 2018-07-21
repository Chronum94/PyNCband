from pyncband import Material, CoreShellParticle
import numpy as np
from numpy import testing as test


def test_type_two_and_he():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc = CoreShellParticle(a, b, 1.0, 1.0)
    assert csnc.type_two  # This is a type 2 CSNC.
    assert csnc.h_e  # This is an h/e structure.


def test_type_two_and_eh():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc = CoreShellParticle(b, a, 1.0, 1.0)
    assert csnc.type_two  # This is a type 2 CSNC.
    assert csnc.e_h  # This is an h/e structure.


def test_type_one():
    a = Material(2.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc = CoreShellParticle(b, a, 1.0, 1.0)
    assert csnc.type_one  # This is a type 1 CSNC.


def test_equal_wavenumbers_for_type_one():
    a = Material(2.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc = CoreShellParticle(b, a, 1.0, 1.0)  # Type 1 CSNC.
    electron_core_wavenumber, electron_shell_wavenumber, hole_core_wavenumber, hole_shell_wavenumber = (
        csnc.calculate_wavenumbers()
    )
    test.assert_allclose(electron_core_wavenumber, hole_core_wavenumber)
    test.assert_allclose(electron_shell_wavenumber, hole_shell_wavenumber)


def test_equal_wavenumbers_for_type_two_eh():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc = CoreShellParticle(a, b, 1.0, 1.0)  # Type 1 CSNC.
    electron_core_wavenumber, electron_shell_wavenumber, hole_core_wavenumber, hole_shell_wavenumber = (
        csnc.calculate_wavenumbers()
    )
    test.assert_allclose(electron_core_wavenumber, hole_shell_wavenumber)
    test.assert_allclose(electron_shell_wavenumber, hole_core_wavenumber)


def test_tighter_confinement_leads_to_higher_energy_states():
    a = Material(2.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(b, a, 1.0, 1.0)  # Type 1 CSNC.
    csnc2 = CoreShellParticle(b, a, 0.9, 1.0)  # Type 1 CSNC.
    csnc1_energies = csnc1.calculate_s1_energies()
    csnc2_energies = csnc2.calculate_s1_energies()
    assert np.all(csnc2_energies > csnc1_energies)


def test_thicker_shell_requires_smaller_core_for_localization_eh():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(b, a, 1.0, 1.0)  # Type 2 CSNC.
    csnc2 = CoreShellParticle(b, a, 1.0, 1.1)  # Type 2 CSNC.
    csnc1_coreloc = csnc1.localization_electron_core()
    csnc2_coreloc = csnc2.localization_electron_core()
    assert csnc2_coreloc < csnc1_coreloc


def test_thicker_shell_requires_smaller_core_for_localization_he():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC, h/e structure.
    csnc2 = CoreShellParticle(a, b, 1.0, 1.1)  # Type 2 CSNC, h/e structure.
    csnc1_coreloc = csnc1.localization_hole_core()
    csnc2_coreloc = csnc2.localization_hole_core()
    assert csnc2_coreloc < csnc1_coreloc


def test_thicker_core_requires_smaller_shell_for_localization_eh():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(b, a, 1.0, 1.0)  # Type 2 CSNC. e/h structure.
    csnc2 = CoreShellParticle(b, a, 1.1, 1.0)  # Type 2 CSNC. e/h structure.
    csnc1_shellloc = csnc1.localization_hole_shell()
    csnc2_shellloc = csnc2.localization_hole_shell()
    assert csnc2_shellloc < csnc1_shellloc


def test_thicker_core_requires_smaller_shell_for_localization_he():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 1.0, 1.0)  # Type 2 CSNC. h/e structure.
    csnc2 = CoreShellParticle(a, b, 1.1, 1.0)  # Type 2 CSNC. h/e structure.
    csnc1_shellloc = csnc1.localization_electron_shell()
    csnc2_shellloc = csnc2.localization_electron_shell()
    assert csnc2_shellloc < csnc1_shellloc


def test_adaptive_energy_bracketing_for_high_energies():
    a = Material(1.0, 0.0, 1.0, 1.0, 1.0)
    b = Material(1.0, -0.5, 1.0, 1.0, 1.0)  # Lower band edges.
    csnc1 = CoreShellParticle(a, b, 0.1,0.1)  # Type 2 CSNC. h/e structure.
    energies = csnc1.calculate_s1_energies()
    assert np.isclose(energies[0], energies[1])


test_adaptive_energy_bracketing_for_high_energies()