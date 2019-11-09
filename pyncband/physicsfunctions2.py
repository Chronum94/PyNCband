import numpy as np
from math import tan
from numba import jit, float64, complex128

from typing import Callable, Union, TYPE_CHECKING, Tuple

__all__ = ["e2k", "k2e", "eigenvalue_residual"]

floatcomplex = Union[float, complex]
floatarray = Union[float, np.ndarray]


def e2k(e, m, potential_step):
    """ Calculates wavenumber from energy.

    Parameters
    ----------
    energy : float, Hartree
        The energy of the state.

    mass : float, electron-masses
        The mass of the electron/hole.

    potential_step : float, Hartree
        If the particle is in a region of nonzero potential energy, then the potential offset.

    Returns
    -------
    wavenumber: float, 1 / Bohr

    """
    return np.sqrt(2 * m * (e - potential_step) + 0.0j)


def k2e(k, m, potential_step):
    """ Calculates energy from wavenumber.

    Parameters
    ----------
    energy : float, Hartree
        The energy of the state.

    mass : float, electron-masses
        The mass of the electron/hole.

    potential_step : float, Hartree
        If the particle is in a region of nonzero potential energy, then the potential offset.

    Returns
    -------
    wavenumber: float, 1 / Bohr

    """
    return np.abs(k) ** 2 / (2 * m) + potential_step


@jit([float64(float64), float64(complex128)], nopython=True, cache=True)
def _tanxdivx(x: floatcomplex) -> float:
    """A custom tan(x)/x function for complex purely real or purely imaginary x, stabilized around |x| = 0,

    Parameters
    ----------
    x : float, complex
        The point at which to evaluate the function.

    Returns
    -------
    val : float
        The real function value at the point.

    """
    # Close to 0, tan(x)/x is just 1, to 15 places of decimal.
    if abs(x) < 1e-8:
        return 1.0
    else:
        return (np.tan(x) / x).real


def eigenvalue_residual(
    energy: float,
    core_radius: float,
    shell_thickness: float,
    core_eff_mass: float,
    shell_eff_mass: float,
    core_potential_step: float,
    shell_potential_step: float,
) -> float:
    """This function returns the residual of the electron energy level eigenvalue equation. Used with root-finding
    methods to calculate the lowest energy state.

    Parameters
    ----------

    energy : float, Hartree
        The energy for which to calculate the wavevector of an electron in the nanoparticle.

    core_radius:

    shell_thickness:

    core_eff_mass:

    shell_eff_mass:

    core_potential_step:

    shell_potential_step:
    References
    ----------

    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

    .. [2] Li, L., Reiss, P., & Protie, M. (2009). Core / Shell Semiconductor Nanocrystals, (2), 154–168. \
        https://doi.org/10.1002/smll.200800841

    """

    core_wavenumber = e2k(energy, core_eff_mass, core_potential_step)
    shell_wavenumber = e2k(energy, shell_eff_mass, shell_potential_step)

    core_x = core_wavenumber * core_radius
    shell_x = shell_wavenumber * shell_thickness
    mass_ratio = shell_eff_mass / core_eff_mass

    return np.real(
        (1.0 - 1.0 / _tanxdivx(core_x)) * mass_ratio - 1.0 - 1.0 / _tanxdivx(shell_x) * core_radius / shell_thickness
    )


@jit(nopython=True)
def _core_wavefunction(x: float, k: floatcomplex, core_radius: float) -> floatcomplex:
    """Returns the value of the S-n pherically symmetric wavefunction in the core.

    Depending on the values of the wavenumber k, will return the radial component of the solution for Schrodinger's
    equation in the core of the semiconductor nanocrystal. Typically used for the lowest k to give the S1 wavefunction
    values.

    Parameters
    ----------
    x : float, bohr
        The position at which to evaluate the wavefunction.

    k : float, purely real or purely imaginary, 1 / bohr
        The wavenumber/momentum of the particle, purely real or purely imaginary.

    core_radius : float, bohr
        The core width of the core-shell quantum dot.

    Returns
    -------
    val : float
        The value of the wavefunction at the point x.

    References
    ----------
    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
    Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
    Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

    """
    denom = np.sin(core_radius * k)

    # The branch is for numerical stability near x = 0.
    if abs(x) < 1e-8:
        val = k / denom
    else:
        val = np.sin(k * x) / (x * denom)
    return val

# core_wavefunction = np.vectorize(_unnormalized_core_wavefunction, otypes=(np.complex128,))

@jit(nopython=True)
def _shell_wavefunction(x: float, q: floatcomplex, core_radius: float, shell_thickness: float) -> floatcomplex:
    """Returns the value of the S1 spherically symmetric wavefunction in the shell.

    Depending on the values of the wavenumber q, will return the radial component of the solution for Schrodinger's
    equation in the shell of the semiconductor nanocrystal. Typically used for the lowest q to give the S1 wavefunction
    values.

    Parameters
    ----------
    x : float, bohr
        The position at which to evaluate the wavefunction.

    q : float, purely real or purely imaginary, 1 / bohr
        The wavenumber/momentum of the particle.

    core_radius : float, bohr
        The core width of the core-shell quantum dot.

    shell_thickness : float, bohr
        The shell thickness of the core-shell quantum dot.

    Returns
    -------
    val : float
        The value of the wavefunction at the point x.

    References
    ----------
    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
    Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
    Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

    """
    # This doesn't need the numerical stability shenanigans because we aren't evaluating it at x = 0.
    # But sin(q * shell_width) can still go to 0, technically. This may not happen because of how q is constrained.
    # Needs testing.
    return np.sin(q * (core_radius + shell_thickness - x)) / (x * np.sin(q * shell_thickness))


@jit(nopython=True)
def _wavefunction(
    x: float, core_wavenumber: floatcomplex, shell_wavenumber: floatcomplex, core_radius: float, shell_thickness: float
) -> floatcomplex:
    """Evaluates the radially symmetric wavefunction values of the core-shell semiconductor nanocrystal at given point.

    A simple wrapper that calls either _core_wavefunction or _shell_wavefunction with core
    wavenumber `k` or shell wavenumber `q`. The `core_radius` and `shell_thickness` variables are obvious.

    Parameters
    ----------

    x : float, bohr
        The radial point at which to evaluate the wavefunction. x can be 0, since the core wavefunction has been
        numerically stabilized at 0.

    core_wavenumber : complex, 1 / bohr
        The (potentially) complex wavevector of the electron/hole in the core of the core-shell particle.

    shell_wavenumber : complex, 1 / bohr
        The (potentially) complex wavevector of the electron/hole in the shell of the core-shell particle.

    core_radius : float, bohr
        The real-valued width of the core of the nanoparticle.

    shell_thickness : float, bohr
        The real-valued width of the shell of the nanoparticle.

    References
    ----------
    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals.
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

    """

    particle_radius = core_radius + shell_thickness

    if 0 <= x < core_radius:
        return _core_wavefunction(x, core_wavenumber, core_radius)
    elif core_radius <= x < particle_radius:
        return _shell_wavefunction(x, shell_wavenumber, core_radius, shell_thickness)
    else:
        return 0