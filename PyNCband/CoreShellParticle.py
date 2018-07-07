
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from .Material import Material
from .physicsfunctions import *

# from functools import partial

__all__ = ['CoreShellParticle']

class CoreShellParticle:
    def __init__(
        self,
        core_material: Material,
        shell_material: Material,
        core_thickness: float,
        shell_thickness: float,
    ):
        self.cmat = core_material
        self.smat = shell_material
        self.core_width = core_thickness
        self.shell_width = shell_thickness

        self.type_one = self.is_type_one()
        self.type_two, self.he, self.eh = self.is_type_two()

        self.ue = np.abs(self.cmat.cbe - self.smat.cbe)
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe)

    # This is likely to get refactored later to return types.
    def is_type_one(self):
        return (self.cmat.vbe > self.smat.vbe) and (self.cmat.cbe < self.smat.cbe)

    def is_type_two(self):
        """"A type two QD has both conduction and valence band edges of its core either higher or lower than the
        corresponding band edges of the shell."""
        core_higher = (self.cmat.vbe > self.smat.vbe) and (
            self.cmat.cbe > self.smat.cbe
        )
        shell_higher = (self.cmat.vbe < self.smat.vbe) and (
            self.cmat.cbe < self.smat.cbe
        )
        return core_higher or shell_higher, core_higher, shell_higher

    def calculate_electron_wavevectors(self, energy: float):
        """Returns a tuple of the electron wavevectors in the core and the shell."""
        result = None
        if self.eh:
            result = (
                wavevector_from_energy(energy, self.cmat.m_e),
                wavevector_from_energy(
                    energy, self.smat.m_e, potential_offset= self.ue
                ),
            )
        elif self.he:
            result = (
                wavevector_from_energy(
                    energy, self.cmat.m_e, potential_offset= self.ue
                ),
                wavevector_from_energy(energy, self.smat.m_e),
            )
        return result

    def calculate_s1_electron_energy(self):
        x = np.linspace(self.ue + 1e-14, self.ue * 10, 1000)
        y = electron_eigenvalue_residual(x, self)
        # print(y)
        y_signs = np.sign(y)
        y_sign_change = np.diff(y_signs) # This array is one element shorter.
        y_neg2pos_change = np.argwhere(np.where(y_sign_change > 0.5, 1, 0))
        root_position = y_neg2pos_change[0]
        # print(*x[root_position:root_position + 2])
        s1_electron_energy = brentq(electron_eigenvalue_residual, x[root_position], x[root_position + 1], args=(self))
        # print(s1_electron_energy)
        # plt.plot(x, y)
        # plt.ylim(-10, 10)
        # plt.show()
        return s1_electron_energy


