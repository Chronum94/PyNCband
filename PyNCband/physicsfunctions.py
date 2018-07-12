import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from numba import jit, vectorize, float64, complex128

from typing import Callable, Union, TYPE_CHECKING

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
    "_wavefunction",
    "wavefunction",
    "make_coulomb_screening_operator",
    "make_interface_polarization_operator",
    "floatcomplex",
]

floatcomplex = Union[float, complex]
floatarray = Union[float, np.ndarray]


@jit(["float64(float64, float64)"], nopython=True)
def _heaviside(x1: float, x2: float) -> float:
    if x1 > 0:
        return 1
    elif x1 == 0:
        return x2
    else:
        return 0


# @vectorize(nopython=True)
@jit(["float64(float64)", "complex128(complex128)"], nopython=True)
def _tanxdivx(x: floatcomplex) -> floatcomplex:
    xsq = x ** 2
    # A simple 2nd order Taylor expansion will be accurate enough this close to 0.
    if abs(x) < 1e-13:
        return 1 - xsq / 2
    else:
        return np.tan(x) / x


tanxdivx = np.vectorize(_tanxdivx)  # np.vectorize(_tanxdivx, otypes=(np.complex128,))

# This is an annoying type signature. I _may_ consider giving this full type signatures, who knows.
@jit(nopython=True)
def _unnormalized_core_wavefunction(
    x: float, k: floatcomplex, core_width: float
) -> floatcomplex:
    ksq = k ** 2  # Useful for the higher powers.
    xsq = x ** 2
    denom = np.sin(core_width * k)

    # The branch is for numerical stability near x = 0.
    if abs(x) < 1e-8:
        # There is no speed penalty for **, so don't try the x * x approach.
        val = 1 / denom * (k - k * ksq * xsq / 6 + k * ksq ** 2 * xsq ** 2 / 120)
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
    # This doesn't need the numerical stability shenanigans because we aren't evaluating it at x = 0.
    # But sin(q * shell_width) can still go to 0, technically. This may not happen because of how q is constrained.
    # Needs testing.
    return np.sin(q * (core_width + shell_width - x)) / (x * np.sin(q * shell_width))


unnormalized_shell_wavefunction = np.vectorize(
    _unnormalized_shell_wavefunction, otypes=(np.complex128,)
)


@jit(nopython=True)
def _wavefunction(
    x: float, k: floatcomplex, q: floatcomplex, core_width: float, shell_width: float
) -> floatcomplex:
    """Evaluates the radially symmetric wavefunction values of the core-shell QD at given points.

    Evaluates the full radial wavefunction of the core-shell quantum dot at given sample points `x`, with core
    wavevector `k` and shell wavevector `q`. The `core_width` and `shell_width` variables are obvious.

    Parameters
    ----------

    x : Array of floats.
        The radial points at which to evaluate the wavefunction. x can contain 0, since the core wavefunction has been
        numerically stabilized at 0.

    k : complex
        The (potentially) complex wavevector of the electron/hole in the core of the core-shell particle.

    q : complex
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
        return _unnormalized_core_wavefunction(xarg, k, core_width)

    def swf(xarg):
        return _unnormalized_shell_wavefunction(xarg, q, core_width, shell_width)

    particle_width = core_width + shell_width

    if 0 <= x < core_width:
        return cwf(x)
    elif core_width <= x < particle_width:
        return swf(x)
    else:
        return 0


# numba.vectorize might be faster, but requires significant refactoring.
wavefunction = np.vectorize(_wavefunction, otypes=(np.complex128,))

# @jit(nopython = True) # Jitting this requires type info for csqrt. need to figure that out.
def wavenumber_from_energy(
    energy: float, mass: float, potential_offset: float = 0
) -> floatcomplex:
    # There's a 1/hbar ** 2 factor under that square root.
    # Omitting it because hbar is obviously 1.
    return csqrt(2 * mass * (energy - potential_offset))


def electron_eigenvalue_residual(
    energy: floatarray, particle: "CoreShellParticle"
) -> float:
    """This function returns the residual of the electron energy level eigenvalue equation. Used with root-finding
    methods to calculate the lowest energy state.

    As of 11-July-2018, this code is not numerically stable if a few tans go to 0. This will be fixed, since the limits
    exist, and they will be conditionally dealt with.

    Parameters
    ----------

    energy : float
        The energy for which to calculate the wavevector of an electron in the nanoparticle.

    particle : CoreShellParticle
        The particle for which to calculate the electron wavevectors. We pass in the particle directly since there
        are a lot of parameters to pass in and this keeps the interface clean.

    References
    ----------

    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals.
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404"""
    k_e, q_e = None, None
    if particle.e_h:
        k_e = wavenumber_from_energy(energy, particle.cmat.m_e)
        q_e = wavenumber_from_energy(
            energy, particle.smat.m_e, potential_offset=particle.ue
        )
    elif particle.h_e:
        k_e = wavenumber_from_energy(
            energy, particle.cmat.m_e, potential_offset=particle.ue
        )
        q_e = wavenumber_from_energy(energy, particle.smat.m_e)
    core_x = k_e * particle.core_width
    shell_x = q_e * particle.shell_width
    core_width = particle.core_width
    shell_width = particle.shell_width
    mass_ratio = particle.smat.m_e / particle.cmat.m_e

    # @jit(nopython = True)
    def _residual():
        return np.real(
            (1 - 1 / tanxdivx(core_x)) * mass_ratio
            - 1
            - 1 / tanxdivx(shell_x) * core_width / shell_width
        )

    return _residual()


def hole_eigenvalue_residual(
    energy: floatarray, particle: "CoreShellParticle"
) -> float:
    """This function returns the residual of the hole energy level eigenvalue equation. Used with root-finding
    methods to calculate the lowest energy state.

    As of 11-July-2018, this code is not numerically stable if a few tans go to 0. This will be fixed, since the limits
    exist, and they will be conditionally dealt with.

    Parameters
    ----------

    energy : float
        The energy for which to calculate the wavevector of a hole in in the nanoparticle.

    particle : CoreShellParticle
        The particle for which to calculate the hole wavevectors. We pass in the particle directly since there
        are a lot of parameters to pass in and this keeps the interface clean.

    References
    ----------

    .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals.
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404"""
    k_h, q_h = None, None
    if particle.e_h:
        k_h = wavenumber_from_energy(energy, particle.cmat.m_e)
        q_h = wavenumber_from_energy(
            energy, particle.smat.m_e, potential_offset=particle.ue
        )
    elif particle.h_e:
        k_h = wavenumber_from_energy(
            energy, particle.cmat.m_e, potential_offset=particle.ue
        )
        q_h = wavenumber_from_energy(energy, particle.smat.m_e)
    core_x = k_h * particle.core_width
    shell_x = q_h * particle.shell_width
    core_width = particle.core_width
    shell_width = particle.shell_width
    mass_ratio = particle.smat.m_h / particle.cmat.m_h

    # @jit(nopython = True)
    def _residual():
        return np.real(
            (1 - 1 / tanxdivx(core_x)) * mass_ratio
            - 1
            - 1 / tanxdivx(shell_x) * core_width / shell_width
        )

    return _residual()


@jit(nopython=True)
def _x_residual_function(x: float, mass_in_core: float, mass_in_shell: float) -> float:
    """This function finds the lower limit for the interval in which to bracket the core localization radius search."""
    m = mass_in_shell / mass_in_core
    xsq = x ** 2
    if abs(x) < 1e-13:
        return m - xsq / 3
    else:
        return 1 / _tanxdivx(x) + m - 1


def make_coulomb_screening_operator(coreshellparticle: "CoreShellParticle") -> Callable:
    core_width = coreshellparticle.core_width
    core_eps, shell_eps = coreshellparticle.cmat.eps, coreshellparticle.smat.eps

    @jit(nopython=True)
    def coulumb_screening_operator(r_a: float, r_b: float) -> float:
        rmax = max(r_a, r_b)
        r_c = core_width
        taz = 0.5  # Theta at zero, theta being step function.
        val = -_heaviside(r_c - r_a, taz) * _heaviside(r_c - r_b, taz) / (
            rmax * core_eps
        ) - (_heaviside(r_a - r_c, taz) + _heaviside(r_b - r_c, taz)) / (
            2 * rmax * shell_eps
        )
        return val

    return coulumb_screening_operator


def make_interface_polarization_operator(
    coreshellparticle: "CoreShellParticle"
) -> Callable:
    core_width = coreshellparticle.core_width
    core_eps, shell_eps = coreshellparticle.cmat.eps, coreshellparticle.smat.eps
    particle_radius = coreshellparticle.radius

    @jit(nopython=True)
    def coulumb_screening_operator(r_a: float, r_b: float) -> float:
        r_c = core_width
        r_p = particle_radius
        taz = 0.5  # Theta at zero, theta being step function.
        val = -_heaviside(r_c - r_a, taz) * _heaviside(r_c - r_b, taz) * (
            core_eps / shell_eps - 1
        ) / (r_c * core_eps) - (shell_eps - 1) / (2 * r_p * shell_eps)
        return val

    return coulumb_screening_operator
