"""This module contains all of the physics and numerics functions required for the implementation of the methods
of the CoreShellparticle and/or Material classes.

"""

from cmath import tan
from typing import Callable, Union, TYPE_CHECKING, Tuple

import numpy as np
from numba import jit, float64, complex128
from numpy.lib.scimath import sqrt as csqrt
from scipy.constants import epsilon_0 as eps0

from .scaling import n_, hbar_ev, m_e, wavenumber_nm_from_energy_ev, e

if TYPE_CHECKING:
    from .CoreShellParticle import CoreShellParticle

__all__ = [
    "unnormalized_core_wavefunction",
    "unnormalized_shell_wavefunction",
    "wavenumber_from_energy",
    "electron_eigenvalue_residual",
    "hole_eigenvalue_residual",
    "minimum_core_localization_size_parameter",
    "_tanxdivx",
    "tanxdivx",
    "_wavefunction",
    "wavefunction",
    "_densityfunction",
    "make_coulomb_screening_operator",
    "make_interface_polarization_operator",
    "make_self_interaction_operator",
    "floatcomplex",
    "scan_and_bracket",
]

floatcomplex = Union[float, complex]
floatarray = Union[float, np.ndarray]


# Using my own Heaviside because Numba doesn't support np.heaviside yet. This may change in the future.
@jit([float64(float64, float64)], nopython=True)
def _heaviside(x1: float, x2: float) -> float:
    """A custom Heaviside function for Numba support.

    Parameters
    ----------
    x1 : float
        The value at which to calculate the function.

    x2 : float
        The value of the function at 0. Typically, this is 0.5 although 0 or 1 are also used.

    Returns
    -------
    val : float
        The value of the Heaviside function at the given point.

    """
    if x1 > 0:
        return 1.0
    elif x1 == 0:
        return x2
    else:
        return 0.0


# @vectorize(nopython=True)
@jit([float64(float64), float64(complex128)], nopython=True)
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
        return (tan(x) / x).real


tanxdivx = np.vectorize(_tanxdivx)  # np.vectorize(_tanxdivx, otypes=(np.complex128,))


# This is an annoying type signature. I _may_ consider giving this full type signatures, who knows.
@jit(nopython=True)
def _unnormalized_core_wavefunction(x: float, k: floatcomplex, core_width: float) -> floatcomplex:
    """Returns the value of the S-n pherically symmetric wavefunction in the core.

    Depending on the values of the wavenumber k, will return the radial component of the solution for Schrodinger's
    equation in the core of the semiconductor nanocrystal. Typically used for the lowest k to give the S1 wavefunction
    values.

    Parameters
    ----------
    x : float, nanometers
        The position at which to evaluate the wavefunction.

    k : float, purely real or purely imaginary, 1 / nm
        The wavenumber/momentum of the particle, purely real or purely imaginary.

    core_width : float, nanometers
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
    denom = np.sin(core_width * k)

    # The branch is for numerical stability near x = 0.
    if abs(x) < 1e-8:
        val = k / denom
    else:
        val = np.sin(k * x) / (x * denom)
    return val


unnormalized_core_wavefunction = np.vectorize(_unnormalized_core_wavefunction, otypes=(np.complex128,))


@jit(nopython=True)
def _unnormalized_shell_wavefunction(x: float, q: floatcomplex, core_width: float, shell_width: float) -> floatcomplex:
    """Returns the value of the S1 spherically symmetric wavefunction in the shell.

    Depending on the values of the wavenumber q, will return the radial component of the solution for Schrodinger's
    equation in the shell of the semiconductor nanocrystal. Typically used for the lowest q to give the S1 wavefunction
    values.

    Parameters
    ----------
    x : float, nanometers
        The position at which to evaluate the wavefunction.

    q : float, purely real or purely imaginary, 1 / nm
        The wavenumber/momentum of the particle.

    core_width : float, nanometers
        The core width of the core-shell quantum dot.

    shell_width : float, nanometers
        The shell width of the core-shell quantum dot.

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
    return np.sin(q * (core_width + shell_width - x)) / (x * np.sin(q * shell_width))


unnormalized_shell_wavefunction = np.vectorize(_unnormalized_shell_wavefunction, otypes=(np.complex128,))


@jit(nopython=True)
def _wavefunction(
    x: float, core_wavenumber: floatcomplex, shell_wavenumber: floatcomplex, core_width: float, shell_width: float
) -> floatcomplex:
    """Evaluates the radially symmetric wavefunction values of the core-shell semiconductor nanocrystal at given point.

    A simple wrapper that calls either __unnormalized_core_wavefunction or __unnormalized_shell_wavefunction with core
    wavevector `k` or shell wavevector `q`. The `core_width` and `shell_width` variables are obvious.

    Parameters
    ----------

    x : float, nanometers
        The radial point at which to evaluate the wavefunction. x can be 0, since the core wavefunction has been
        numerically stabilized at 0.

    core_wavenumber : complex, 1 / nm
        The (potentially) complex wavevector of the electron/hole in the core of the core-shell particle.

    shell_wavenumber : complex, 1 / nm
        The (potentially) complex wavevector of the electron/hole in the shell of the core-shell particle.

    core_width : float, nanometers
        The real-valued width of the core of the nanoparticle.

    shell_width : float, nanometers
        The real-valued width of the shell of the nanoparticle.

    References
    ----------
    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals.
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

    """

    particle_width = core_width + shell_width

    if 0 <= x < core_width:
        return _unnormalized_core_wavefunction(x, core_wavenumber, core_width)
    elif core_width <= x < particle_width:
        return _unnormalized_shell_wavefunction(x, shell_wavenumber, core_width, shell_width)
    else:
        return 0


# numba.vectorize might be faster, but requires significant refactoring.
wavefunction = np.vectorize(_wavefunction, otypes=(np.complex128,))


@jit(nopython=True)
def _densityfunction(
    r: float, core_wavenumber: floatcomplex, shell_wavenumber: floatcomplex, core_width: float, shell_width: float
) -> float:
    """Returns the probability density from a wavefunction at a point in the core-shell.


    Parameters
    ----------
    r : float, nanometers
        The point at which to evaluate the probability density.

    core_wavenumber : float, purely real or purely imaginary, 1 / nanometers
        The wavenumber in the core of the core-shell quantum dot.

    shell_wavenumber : float, purely real or purely imaginary, 1 / nanometers
        The wavenumber in the shell of the core-shell quantum dot.

    core_width : float, nanometers
        The width of the core of the quantum dot.

    shell_width : float, nanometers
        The width of the shell of the quantum dot.
    Returns
    -------
    val : float
        The probabilty density of the partcle at that radial point in the core-shell semiconductor nanocrystal.

    """
    return abs(_wavefunction(r, core_wavenumber, shell_wavenumber, core_width, shell_width)) ** 2


# @jit(nopython = True) # Jitting this requires type info for csqrt. need to figure that out.
def wavenumber_from_energy(energy: float, mass: float, potential_offset: float = 0) -> floatcomplex:
    """ Calculates wavenumber from energy.

    Parameters
    ----------
    energy : float, eV
        The energy of the state.

    mass : float, electron-masses
        The mass of the particle.

    potential_offset : float, eV
        If the particle is in a region of nonzero potential energy, then the potential offset.

    Returns
    -------
    wavenumber: float, 1 / nm

    """

    # The 2.498... is to convert to 1/nm when using eV and hbar in eV-s together.
    return csqrt(2 * mass * m_e * (energy - potential_offset)) / hbar_ev * wavenumber_nm_from_energy_ev


def electron_eigenvalue_residual(energy: floatarray, particle: "CoreShellParticle") -> float:
    """This function returns the residual of the electron energy level eigenvalue equation. Used with root-finding
    methods to calculate the lowest energy state.

    Parameters
    ----------

    energy : float, eV
        The energy for which to calculate the wavevector of an electron in the nanoparticle.

    particle : CoreShellParticle
        The particle for which to calculate the electron wavevectors. We pass in the particle directly since there
        are a lot of parameters to pass in and this keeps the interface clean.

    References
    ----------

    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

    .. [2] Li, L., Reiss, P., & Protie, M. (2009). Core / Shell Semiconductor Nanocrystals, (2), 154–168. \
        https://doi.org/10.1002/smll.200800841

    """
    core_electron_wavenumber, shell_electron_wavenumber = (None, None)

    if particle.type_one:
        # Energy step is in the shell.
        core_electron_wavenumber = wavenumber_from_energy(energy, particle.cmat.m_e)
        shell_electron_wavenumber = wavenumber_from_energy(energy, particle.smat.m_e, potential_offset=particle.ue)

    elif particle.type_one_reverse:
        # Energy step is in the core.
        core_electron_wavenumber = wavenumber_from_energy(energy, particle.cmat.m_e, potential_offset=particle.ue)
        shell_electron_wavenumber = wavenumber_from_energy(energy, particle.smat.m_e)

    elif particle.type_two:
        if particle.e_h:
            core_electron_wavenumber = wavenumber_from_energy(energy, particle.cmat.m_e)
            shell_electron_wavenumber = wavenumber_from_energy(energy, particle.smat.m_e, potential_offset=particle.ue)
        elif particle.h_e:
            core_electron_wavenumber = wavenumber_from_energy(energy, particle.cmat.m_e, potential_offset=particle.ue)
            shell_electron_wavenumber = wavenumber_from_energy(energy, particle.smat.m_e)
    core_x = core_electron_wavenumber * particle.core_width
    shell_x = shell_electron_wavenumber * particle.shell_width
    core_width = particle.core_width
    shell_width = particle.shell_width
    mass_ratio = particle.smat.m_e / particle.cmat.m_e

    if type(core_x) in [np.float64, np.complex128]:
        return np.real((1 - 1 / _tanxdivx(core_x)) * mass_ratio - 1 - 1 / _tanxdivx(shell_x) * core_width / shell_width)
    else:
        return np.real((1 - 1 / tanxdivx(core_x)) * mass_ratio - 1 - 1 / tanxdivx(shell_x) * core_width / shell_width)


def hole_eigenvalue_residual(energy: floatarray, particle: "CoreShellParticle") -> float:
    """This function returns the residual of the hole energy level eigenvalue equation. Used with root-finding
    methods to calculate the lowest energy state.

    Parameters
    ----------

    energy : float, eV
        The energy for which to calculate the wavevector of a hole in in the nanoparticle.

    particle : CoreShellParticle
        The particle for which to calculate the hole wavevectors. We pass in the particle directly since there
        are a lot of parameters to pass in and this keeps the interface clean.

    References
    ----------

    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals.
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

    .. [2] Li, L., Reiss, P., & Protie, M. (2009). Core / Shell Semiconductor Nanocrystals, (2), 154–168.
        https://doi.org/10.1002/smll.200800841

    """
    core_hole_wavenumber, shell_hole_wavenumber = (None, None)
    if particle.type_one:
        core_hole_wavenumber = wavenumber_from_energy(energy, particle.cmat.m_h)
        shell_hole_wavenumber = wavenumber_from_energy(energy, particle.smat.m_h, potential_offset=particle.uh)

    elif particle.type_one_reverse:
        core_hole_wavenumber = wavenumber_from_energy(energy, particle.cmat.m_h, potential_offset=particle.uh)
        shell_hole_wavenumber = wavenumber_from_energy(energy, particle.smat.m_h)

    elif particle.type_two:
        if particle.e_h:
            core_hole_wavenumber = wavenumber_from_energy(energy, particle.cmat.m_h, potential_offset=particle.uh)
            shell_hole_wavenumber = wavenumber_from_energy(energy, particle.smat.m_h)
        elif particle.h_e:
            core_hole_wavenumber = wavenumber_from_energy(energy, particle.cmat.m_h)
            shell_hole_wavenumber = wavenumber_from_energy(energy, particle.smat.m_h, potential_offset=particle.uh)
    core_x = core_hole_wavenumber * particle.core_width
    shell_x = shell_hole_wavenumber * particle.shell_width
    core_width = particle.core_width
    shell_width = particle.shell_width
    mass_ratio = particle.smat.m_h / particle.cmat.m_h

    if type(core_x) in [np.float64, np.complex128]:
        return np.real((1 - 1 / _tanxdivx(core_x)) * mass_ratio - 1 - 1 / _tanxdivx(shell_x) * core_width / shell_width)
    else:
        return np.real((1 - 1 / tanxdivx(core_x)) * mass_ratio - 1 - 1 / tanxdivx(shell_x) * core_width / shell_width)


@jit(nopython=True)
def minimum_core_localization_size_parameter(x: float, mass_in_core: float, mass_in_shell: float) -> float:
    """This function finds the lower limit for the interval in which to bracket the core localization radius search.

    Parameters
    ----------
    x : float
        Size parameter (wavenumber * core radius) in the core of the core-shell quantum dor. From equation 3 in [1].

    mass_in_core : float, electron-masses
        Effective mass of the electron/hole in the core of the core-shell quantum dot.

    mass_in_shell : float, electron-masses
        Effective mass of the electron/hole in the shell of the core-shell quantum dot.

    Returns
    -------
    residual : float
        The residual of the equation, used in root-finding routines.

    References
    ----------
    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
    Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
    Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

    """
    mass_ratio_shellcore = mass_in_shell / mass_in_core
    if abs(x) < 1e-8:
        return 1 / mass_ratio_shellcore
    else:
        return 1 / _tanxdivx(x) + 1 / mass_ratio_shellcore - 1


def make_coulomb_screening_operator(coreshellparticle: "CoreShellParticle") -> Callable:
    """Creates a Coulomb interaction operator function for a particular function.

    Currently, this assumes opposite charges on the two interacting particles. This will likely need to be changed once
    biexcitons are considered.

    Parameters
    ----------
    coreshellparticle : CoreShellParticle

    Returns
    -------
    coulomb_screening_operator : Callable(r1, r2)
        The Coulomb screening operator as a function of the two radial coordinates of the two excitons.

    References
    ----------
    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
    Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
    Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

    """

    # Stripping these variables of the coreshellparticle so Numba can use them.
    core_width = coreshellparticle.core_width
    core_eps, shell_eps = (coreshellparticle.cmat.eps, coreshellparticle.smat.eps)

    @jit([float64(float64, float64)], nopython=True)
    def coulomb_screening_operator(r_a: float, r_b: float) -> float:
        rmax = max(r_a, r_b)
        r_c = core_width
        taz = 0.5  # Theta at zero, theta being step function.

        # The two step functions that are used to calculate the charge regions in the Coulomb interaction operator.
        step1, step2 = (_heaviside(r_c - r_a, taz), _heaviside(r_c - r_b, taz))
        val = -step1 * step2 / (rmax * core_eps) - (1 - step1 + 1 - step2) / (2 * rmax * shell_eps)
        return val * e / n_ * 1 / (4.0 * np.pi * eps0)  # Scaling to eV.

    return coulomb_screening_operator


def make_interface_polarization_operator(coreshellparticle: "CoreShellParticle") -> Callable:
    """Generates the interface polarization operator from the CoreShellParticle inforamtion.

    Parameters
    ----------
    coreshellparticle : CoreShellParticle

    Returns
    -------
    coulomb_screening_operator : Callable(r1, r2)
        The interface polarization operator as a function of the two radial coordinates of the two excitons.
    """

    # Stripping variables of coreshellparticle so Numba can use them.
    core_width = coreshellparticle.core_width
    core_eps, shell_eps = (coreshellparticle.cmat.eps, coreshellparticle.smat.eps)
    particle_radius = coreshellparticle.radius
    env_eps = coreshellparticle.environment_epsilon

    @jit(nopython=True)
    def interface_polarization_operator(r_a: float, r_b: float) -> float:
        r_c = core_width
        r_p = particle_radius
        taz = 0.5  # Theta at zero, theta being step function.
        val = -_heaviside(r_c - r_a, taz) * _heaviside(r_c - r_b, taz) * (core_eps / shell_eps - 1) / (
            r_c * core_eps
        ) - (shell_eps / env_eps - 1) / (2 * r_p * shell_eps)
        return val * e / (n_ * eps0) * 1 / (4.0 * np.pi)  # Scaling with physical quantities.

    return interface_polarization_operator


def make_self_interaction_operator(coreshellparticle: "CoreShellParticle") -> Callable:
    """Creates a self-interaction operator function for a particular particle.

    This is the operator to model the respulsion that one part of an electron/hole's wavefunction experiences due to \
    the other parts of the wavefunction of the same particle.

    Parameters
    ----------
    coreshellparticle : CoreShellParticle

    Returns
    -------
    self_interaction_operator : Callable(r1, r2)
        The Coulomb screening operator as a function of the two radial coordinates of the two excitons.

    References
    ----------
    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
    Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
    Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

    """

    # Stripping these variables of the coreshellparticle so Numba can use them.
    core_width = coreshellparticle.core_width
    core_eps, shell_eps = (coreshellparticle.cmat.eps, coreshellparticle.smat.eps)
    particle_radius = coreshellparticle.radius
    env_eps = coreshellparticle.environment_epsilon

    @jit([float64(float64)], nopython=True)
    def self_interaction_operator(r) -> float:

        MAX_EXPANSION_ORDER = 150
        r_c = core_width
        r_p = particle_radius
        operator_value = 0.0
        screening_multiplier = None
        rbar = r_c / r_p

        # Checked times: 6.
        if r < r_c:
            operator_value = 0.0
            for n in range(MAX_EXPANSION_ORDER):
                # print("LMAOCORE")
                eps_cs_neg = core_eps / shell_eps - 1
                eps_cs_pos = eps_cs_neg + 2

                eps_se_neg = shell_eps / env_eps - 1
                eps_se_pos = eps_se_neg + 2

                alpha_n = n * eps_cs_neg / (n * eps_cs_pos + 1)
                beta_n = (n + 1) * eps_se_neg / (n * eps_se_pos + 1)
                gamma_n = (2 * n + 1) / (n * eps_cs_pos + 1)

                alpha_beta_pos = 1 + alpha_n * beta_n * rbar ** (2 * n + 1)

                core_term = (
                    (gamma_n * eps_cs_neg / alpha_beta_pos)
                    * r ** (2 * n)
                    / r_c ** (2 * n + 1)
                )

                shell_term = (
                    (beta_n * (gamma_n * core_eps / shell_eps - alpha_n) / alpha_beta_pos)
                    * r ** (2 * n)
                    / r_p ** (2 * n + 1)
                )

                operator_value += core_term + shell_term
            screening_multiplier = 0.5 / core_eps
        elif r >= r_c:
            operator_value = 0.0
            for n in range(MAX_EXPANSION_ORDER):
                # print("LMAOSHELL")
                eps_cs_neg = core_eps / shell_eps - 1
                eps_cs_pos = eps_cs_neg + 2

                eps_se_neg = shell_eps / env_eps - 1
                eps_se_pos = eps_se_neg + 2

                alpha_n = n * eps_cs_neg / (n * eps_cs_pos + 1)
                beta_n = (n + 1) * eps_se_neg / (n * eps_se_pos + 1)

                alpha_beta_pos = 1 + alpha_n * beta_n * rbar ** (2 * n + 1)

                shell_term = (beta_n / alpha_beta_pos) * r ** (2 * n) / r_p ** (2 * n + 1)

                env_term_one = (
                    -2 * (alpha_beta_pos - 1) / (alpha_beta_pos * r)
                )

                env_term_two = (
                    -(alpha_n / alpha_beta_pos) * r_c ** (2 * n + 1) / r ** (2 * n + 2)
                )

                operator_value += shell_term + env_term_one + env_term_two

            screening_multiplier = 0.5 / shell_eps
        return operator_value * e / (n_ * eps0) * 1 / (4.0 * np.pi) * screening_multiplier

    return self_interaction_operator


def scan_and_bracket(
    f: Callable, lower_bound: float, upper_bound: float, resolution: int
) -> Tuple[Tuple[float, float], bool]:
    """Attempts to scan for roots and bracket roots of a function where the function goes from negative to positive.

    Parameters
    ----------
    f : Callable
        A function that accepts only one argument. Use lambdas if it has args.
    lower_bound : float
        The lower bound of the scan.
    upper_bound : float
        The upper bound of the scan.
    resolution : int
        The number of points in the scanning interval.

    Returns
    -------
    bracket : Tuple of floats

    bracket_found : bool
        If a valid bracket was found.
    """
    x = np.linspace(lower_bound, upper_bound, resolution)
    y = f(x)

    y_signs = np.sign(y)
    y_sign_change = np.diff(y_signs)  # This array is one element shorter.

    # The 0.5 thresholding is mostly arbitrary. A 0 would work just fine
    y_neg2pos_change = np.argwhere(np.where(y_sign_change > 0.5, 1, 0))

    # If a bracket has been found, send bracket limits and a bracket-found boolean.
    if y_neg2pos_change.shape[0] > 0:
        root_position = y_neg2pos_change[0]
        return (x[root_position], x[root_position + 1]), True

    # In case a bracket is not found, send in dummy bracket limits, and the bool.
    else:
        return (0.0, 0.0), False
