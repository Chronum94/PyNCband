import numpy as np
from numpy.lib.scimath import sqrt as csqrt


def core_wavefunction(x, k: float, core_width: float):
    return np.sin(k * x) / (x * np.sin(k * core_width))


def shell_wavefunction(x, k: float, core_width: float, shell_width: float):
    return np.sin(k * (core_width + shell_width - x)) / (x * np.sin(k * shell_width))


def wavevector_from_energy(energy: float, mass: float, potential_offset: float = 0):
    # There's a 1/hbar ** 2 factor under that square root.
    # Omitting it because hbar is obviously 1.
    return csqrt(2 * mass * (energy - potential_offset))
