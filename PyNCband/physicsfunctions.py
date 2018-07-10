import numpy as np
from numpy.lib.scimath import sqrt as csqrt

# from .CoreShellParticle import CoreShellParticle

__all__ = [
    "unnormalized_core_wavefunction",
    "unnormalized_shell_wavefunction",
    "wavevector_from_energy",
    "electron_eigenvalue_residual",
    "hole_eigenvalue_residual",
    "_x_residual_function"
]


def unnormalized_core_wavefunction(x, k: float, core_width: float):
    return np.sin(k * x) / (x * np.sin(k * core_width))


def unnormalized_shell_wavefunction(x, q: float, core_width: float, shell_width: float):
    return np.sin(q * (core_width + shell_width - x)) / (x * np.sin(q * shell_width))

def wavefunction(x, k: float, q: float, core_width: float, shell_width: float):
    if 0 <= x < core_width:
        val = unnormalized_core_wavefunction(x, k, core_width)
    elif core_width <= x < core_width + shell_width:
        val = unnormalized_shell_wavefunction(x, q, core_width, shell_width)
    else:
        val = 0
    return val



def wavevector_from_energy(energy: float, mass: float, potential_offset: float = 0):
    # There's a 1/hbar ** 2 factor under that square root.
    # Omitting it because hbar is obviously 1.
    return csqrt(2 * mass * (energy - potential_offset))


def electron_eigenvalue_residual(energy, particle):
    if particle.e_h:
        k_e = wavevector_from_energy(energy, particle.cmat.m_e)
        q_e = wavevector_from_energy(
            energy, particle.smat.m_e, potential_offset=particle.ue
        )
        core_x = k_e * particle.core_width
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                (1 - core_x / np.tan(core_x)) * particle.smat.m_e / particle.cmat.m_e
                - 1
                - q_e * particle.core_width / np.tan(q_e * particle.shell_width)
            )


def hole_eigenvalue_residual(energy, particle):
    if particle.e_h:
        k_h = wavevector_from_energy(
            energy, particle.cmat.m_h, potential_offset=particle.uh
        )
        q_h = wavevector_from_energy(energy, particle.smat.m_h)

        core_x = k_h * particle.core_width
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                (1 - core_x / np.tan(core_x)) * particle.smat.m_h / particle.cmat.m_h
                - 1
                - q_h * particle.core_width / np.tan(q_h * particle.shell_width)
            )

def _x_residual_function(x, mass_in_core: float, mass_in_shell: float):
        m = mass_in_shell / mass_in_core
        xsq = x ** 2
        if - 1e-3 < x < 1e-3:
            return m - xsq / 3 - xsq ** 2 / 45 - xsq ** 3 / 945 # Somewhat warried about floating point round-off.
        else:
            return x / np.tan(x) + m - 1
