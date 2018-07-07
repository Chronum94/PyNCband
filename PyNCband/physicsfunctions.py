import numpy as np
from numpy.lib.scimath import sqrt as csqrt

# from .CoreShellParticle import CoreShellParticle

__all__ = [
    "core_wavefunction",
    "shell_wavefunction",
    "wavevector_from_energy",
    "electron_eigenvalue_residual",
    "hole_eigenvalue_residual",
]


def core_wavefunction(x, k: float, core_width: float):
    return np.sin(k * x) / (x * np.sin(k * core_width))


def shell_wavefunction(x, k: float, core_width: float, shell_width: float):
    return np.sin(k * (core_width + shell_width - x)) / (x * np.sin(k * shell_width))


def wavevector_from_energy(energy: float, mass: float, potential_offset: float = 0):
    # There's a 1/hbar ** 2 factor under that square root.
    # Omitting it because hbar is obviously 1.
    return csqrt(2 * mass * (energy - potential_offset))


def electron_eigenvalue_residual(energy, particle):
    if particle.eh:
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
    if particle.eh:
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
