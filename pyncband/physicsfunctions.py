from cmath import tan

import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from numba import jit, float64, complex128

from scipy.constants import hbar, e, m_e, epsilon_0 as eps0

from .scaling import n_
from .utils import EnergyNotBracketedError
from typing import Callable, Union, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .CoreShellParticle import CoreShellParticle

# This only imports the  CoreShellFunction class and is done because otherwise the cyclic import fails.

__all__ = [
    "unnormalized_core_wavefunction",
    "unnormalized_shell_wavefunction",
    "wavenumber_from_energy",
    "electron_eigenvalue_residual",
    "hole_eigenvalue_residual",
    "_x_residual_function",
    "_tanxdivx",
    "tanxdivx",
    "_wavefunction",
    "wavefunction",
    "_densityfunction",
    "make_coulomb_screening_operator",
    "make_interface_polarization_operator",
    "floatcomplex",
    "scan_and_bracket",
]

floatcomplex = Union[float, complex]
floatarray = Union[float, np.ndarray]


@jit([float64(float64, float64)], nopython=True)
def _heaviside(x1: float, x2: float) -> float:
    """A custom Heaviside function for number support.

    Parameters
    ----------
    x1 : float
        The value at which to calculate the function.

    x2 : float
        The value of the function at 0. Typically, this is 0.5 although 0 or 1 are also used.

    Returns
    -------
    float : The value of the Heaviside function at the given point.

    """
    if x1 > 0:
        return 1.0
    elif x1 == 0:
        return x2
    else:
        return 0.0


# @vectorize(nopython=True)
@jit([float64(float64), float64(complex128)], nopython=True)
def _tanxdivx(x: floatcomplex) -> floatcomplex:
    """A custom tan(x)/x function for complex purely real or purely imaginary x, stabilized around |x| = 0,

    Parameters
    ----------
    x : float, complex
        The point at which to evaluate the function.

    Returns
    -------
    float : The real function value at the point.

    """
    # Close to 0, tan(x)/x is just 1, to 15 places of decimal.
    if abs(x) < 1e-10:
        return 1.0
    else:
        return (tan(x) / x).real


tanxdivx = np.vectorize(_tanxdivx)  # np.vectorize(_tanxdivx, otypes=(np.complex128,))

# This is an annoying type signature. I _may_ consider giving this full type signatures, who knows.
@jit(nopython=True)
def _unnormalized_core_wavefunction(
    x: float, k: floatcomplex, core_width: float
) -> floatcomplex:
    """Returns the value of the S1 spherically symmetric wavefunction in the core.

    Parameters
    ----------
    x : float, nanometers
        The position at which to evaluate the wavefunction.

    k : float, 1 / nm
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
        # There is no speed penalty for **, so don't try the x * x approach.
        val = k / denom
    else:
        val = np.sin(k * x) / (x * denom)
    return val


unnormalized_core_wavefunction = np.vectorize(
    _unnormalized_core_wavefunction, otypes=(np.complex128,)
)


@jit(nopython=True)
def _unnormalized_shell_wavefunction(
    x: float, q: floatcomplex, core_width: float, shell_width: float
) -> floatcomplex:
    """Returns the value of the S1 spherically symmetric wavefunction in the shell.

    Parameters
    ----------
    x : float
        The position at which to evaluate the wavefunction.

    q : float, purely real or purely imaginary
        The wavenumber/momentum of the particle.

    core_width : float
        The core width of the core-shell quantum dot.

    shell_width : float
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


unnormalized_shell_wavefunction = np.vectorize(
    _unnormalized_shell_wavefunction, otypes=(np.complex128,)
)


@jit(nopython=True)
def _wavefunction(
    x: float,
    core_wavenumber: floatcomplex,
    shell_wavenumber: floatcomplex,
    core_width: float,
    shell_width: float,
) -> floatcomplex:
    """Evaluates the radially symmetric wavefunction values of the core-shell QD at given points.

    Evaluates the full radial wavefunction of the core-shell quantum dot at given sample points `x`, with core
    wavevector `k` and shell wavevector `q`. The `core_width` and `shell_width` variables are obvious.

    Parameters
    ----------

    x : float
        The radial point at which to evaluate the wavefunction. x can contain 0, since the core wavefunction has been
        numerically stabilized at 0.

    core_wavenumber : complex
        The (potentially) complex wavevector of the electron/hole in the core of the core-shell particle.

    shell_wavenumber : complex
        The (potentially) complex wavevector of the electron/hole in the shell of the core-shell particle.

    core_width : float
        The real-valued width of the core of the nanoparticle.

    shell_width : float
        The real-valued width of the shell of the nanoparticle.

    References
    ----------
    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals.
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404"""

    def cwf(xarg):
        return _unnormalized_core_wavefunction(xarg, core_wavenumber, core_width)

    def swf(xarg):
        return _unnormalized_shell_wavefunction(
            xarg, shell_wavenumber, core_width, shell_width
        )

    particle_width = core_width + shell_width

    if 0 <= x < core_width:
        return cwf(x)
    elif core_width <= x < particle_width:
        return swf(x)
    else:
        return 0


# numba.vectorize might be faster, but requires significant refactoring.
wavefunction = np.vectorize(_wavefunction, otypes=(np.complex128,))


@jit(nopython=True)
def _densityfunction(
    r: float,
    core_wavenumber: floatcomplex,
    shell_wavenumber: floatcomplex,
    core_width: float,
    shell_width: float,
) -> float:
    """Returns the probability density from a wavefunction at a point in the core-shell.

    Parameters
    ----------
    r : float
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

    """
    return (
        abs(
            _wavefunction(r, core_wavenumber, shell_wavenumber, core_width, shell_width)
        )
        ** 2
    )


# @jit(nopython = True) # Jitting this requires type info for csqrt. need to figure that out.
def wavenumber_from_energy(
    energy: float, mass: float, potential_offset: float = 0
) -> floatcomplex:
    """ Calculates wavenumber from energy in units of 1/m.

    Parameters
    ----------
    energy : float, Joules
    mass : float, electron-masses
    potential_offset : float, Joules

    Returns
    -------
    wavenumber: float 1 / m
    """

    return csqrt(2 * mass * m_e * (energy - potential_offset)) / hbar


def electron_eigenvalue_residual(
    energy: floatarray, particle: "CoreShellParticle"
) -> float:
    """This function returns the residual of the electron energy level eigenvalue equation. Used with root-finding
    methods to calculate the lowest energy state.

    As of 11-July-2018, this code is not numerically stable if a few tans go to 0. This will be fixed, since the limits
    exist, and they will be conditionally dealt with.

    Parameters
    ----------

    energy : float, Joules
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
    core_electron_wavenumber, shell_electron_wavenumber = None, None

    if particle.type_one:
        # Energy step is in the core.
        core_electron_wavenumber = (
            wavenumber_from_energy(
                energy, particle.cmat.m_e, potential_offset=particle.ue
            )
            * n_
        )
        shell_electron_wavenumber = (
            wavenumber_from_energy(energy, particle.smat.m_e) * n_
        )

    elif particle.type_two:
        if particle.e_h:
            core_electron_wavenumber = (
                wavenumber_from_energy(energy, particle.cmat.m_e) * n_
            )
            shell_electron_wavenumber = (
                wavenumber_from_energy(
                    energy, particle.smat.m_e, potential_offset=particle.ue
                )
                * n_
            )
        elif particle.h_e:
            core_electron_wavenumber = (
                wavenumber_from_energy(
                    energy, particle.cmat.m_e, potential_offset=particle.ue
                )
                * n_
            )
            shell_electron_wavenumber = (
                wavenumber_from_energy(energy, particle.smat.m_e) * n_
            )
    core_x = core_electron_wavenumber * particle.core_width
    shell_x = shell_electron_wavenumber * particle.shell_width
    core_width = particle.core_width
    shell_width = particle.shell_width
    mass_ratio = particle.smat.m_e / particle.cmat.m_e

    # shelltan = 1 / tanxdivx(shell_x)
    # coretan = 1 / tanxdivx(core_x)
    # if type(shelltan) not in [np.float64, np.complex128]:
    #     a = max(np.imag(shell_x))
    #     b = max(np.imag(core_x))
    # print("Something large:", np.any(np.imag(shell_x) > 1e4))
    # print("Something large:", np.any(np.imag(core_x) > 1e4))
    if type(core_x) in [np.float64, np.complex128]:
        return np.real(
            (1 - 1 / _tanxdivx(core_x)) * mass_ratio
            - 1
            - 1 / _tanxdivx(shell_x) * core_width / shell_width
        )
    else:
        return np.real(
            (1 - 1 / tanxdivx(core_x)) * mass_ratio
            - 1
            - 1 / tanxdivx(shell_x) * core_width / shell_width
        )


def hole_eigenvalue_residual(
    energy: floatarray, particle: "CoreShellParticle"
) -> float:
    """This function returns the residual of the hole energy level eigenvalue equation. Used with root-finding
    methods to calculate the lowest energy state.

    As of 11-July-2018, this code is not numerically stable if a few tans go to 0. This will be fixed, since the limits
    exist, and they will be conditionally dealt with.

    Parameters
    ----------

    energy : float, Joules
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
    core_hole_wavenumber, shell_hole_wavenumber = None, None
    if particle.type_one:
        core_hole_wavenumber = (
            wavenumber_from_energy(
                energy, particle.cmat.m_h, potential_offset=particle.uh
            )
            * n_
        )
        shell_hole_wavenumber = wavenumber_from_energy(energy, particle.smat.m_h) * n_

    elif particle.type_two:
        if particle.e_h:
            core_hole_wavenumber = (
                wavenumber_from_energy(
                    energy, particle.cmat.m_h, potential_offset=particle.uh
                )
                * n_
            )
            shell_hole_wavenumber = (
                wavenumber_from_energy(energy, particle.smat.m_h) * n_
            )
        elif particle.h_e:
            core_hole_wavenumber = (
                wavenumber_from_energy(energy, particle.cmat.m_h) * n_
            )
            shell_hole_wavenumber = (
                wavenumber_from_energy(
                    energy, particle.smat.m_h, potential_offset=particle.uh
                )
                * n_
            )
    core_x = core_hole_wavenumber * particle.core_width
    shell_x = shell_hole_wavenumber * particle.shell_width
    core_width = particle.core_width
    shell_width = particle.shell_width
    mass_ratio = particle.smat.m_h / particle.cmat.m_h

    if type(core_x) in [np.float64, np.complex128]:
        return np.real(
            (1 - 1 / _tanxdivx(core_x)) * mass_ratio
            - 1
            - 1 / _tanxdivx(shell_x) * core_width / shell_width
        )
    else:
        return np.real(
            (1 - 1 / tanxdivx(core_x)) * mass_ratio
            - 1
            - 1 / tanxdivx(shell_x) * core_width / shell_width
        )


@jit(nopython=True)
def _x_residual_function(x: float, mass_in_core: float, mass_in_shell: float) -> float:
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
    mass_ratio = mass_in_shell / mass_in_core
    if abs(x) < 1e-10:
        return mass_ratio
    else:
        return 1 / _tanxdivx(x) + mass_ratio - 1


def make_coulomb_screening_operator(coreshellparticle: "CoreShellParticle") -> Callable:
    """

    Parameters
    ----------
    coreshellparticle

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

    core_width = coreshellparticle.core_width
    core_eps, shell_eps = coreshellparticle.cmat.eps, coreshellparticle.smat.eps

    @jit([float64(float64, float64)], nopython=True)
    def coulomb_screening_operator(r_a: float, r_b: float) -> float:
        rmax = max(r_a, r_b)
        r_c = core_width
        taz = 1.0  # Theta at zero, theta being step function.

        # The two step functions that are used to calculate the charge regions in the Coulomb interaction operator.
        step1, step2 = _heaviside(r_c - r_a, taz), _heaviside(r_c - r_b, taz)
        val = -step1 * step2 / (rmax * core_eps) - (1 - step1 + 1 - step2) / (
            2 * rmax * shell_eps
        )
        return val * e / (n_ * eps0) * 1 / (4.0 * np.pi)  # Scaling to eV and meters.

    return coulomb_screening_operator


def make_interface_polarization_operator(
    coreshellparticle: "CoreShellParticle"
) -> Callable:
    """Generates the interface polarization operator from the CoreShellParticle inforamtion.

    Parameters
    ----------
    coreshellparticle

    Returns
    -------

    """

    # Scaling lengths to nm units.
    core_width = coreshellparticle.core_width
    core_eps, shell_eps = coreshellparticle.cmat.eps, coreshellparticle.smat.eps
    particle_radius = coreshellparticle.radius
    env_eps = coreshellparticle.environment_epsilon

    @jit(nopython=True)
    def interface_polarization_operator(r_a: float, r_b: float) -> float:
        r_c = core_width
        r_p = particle_radius
        taz = 1.0  # Theta at zero, theta being step function.
        val = -_heaviside(r_c - r_a, taz) * _heaviside(r_c - r_b, taz) * (
            core_eps / shell_eps - 1
        ) / (r_c * core_eps) - (shell_eps / env_eps - 1) / (2 * r_p * shell_eps)
        return (
            val * e / (n_ * eps0) * 1 / (4.0 * np.pi)
        )  # Scaling with physical quantities.

    return interface_polarization_operator


def scan_and_bracket(
    f: Callable, lower_bound: float, upper_bound: float, resolution: int
) -> Tuple[float, float]:
    """ Attempts to scan for roots and bracket roots of a function where the function goes from negative to positive.

    Parameters
    ----------
    f
    lower_bound
    upper_bound
    resolution

    Returns
    -------

    """
    x = np.linspace(lower_bound, upper_bound, resolution)
    y = f(x)

    y_signs = np.sign(y)
    y_sign_change = np.diff(y_signs)  # This array is one element shorter.

    # The 0.5 thresholding is mostly arbitrary. A 0 would work just fine
    y_neg2pos_change = np.argwhere(np.where(y_sign_change > 0.5, 1, 0))
    if y_neg2pos_change.shape[0] > 0:
        root_position = y_neg2pos_change[0]
        return x[root_position], x[root_position + 1]
    else:
        raise EnergyNotBracketedError(
            "Try increasing the upper energy bound to bracket the energy of the first state."
        )
