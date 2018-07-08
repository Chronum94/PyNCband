
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from .Material import Material
from .physicsfunctions import *

# from functools import partial

__all__ = ["CoreShellParticle"]


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
        self.radius = core_thickness + shell_thickness
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

    def calculate_electron_wavevectors(self):
        """Returns a tuple of the electron wavevectors in the core and the shell."""
        energy_e, energy_h = self.calculate_s1_energies()
        result = None
        if self.eh:
            result = (
                wavevector_from_energy(energy_e, self.cmat.m_e),
                wavevector_from_energy(
                    energy_e, self.smat.m_e, potential_offset=self.ue
                ),
            )
        elif self.he:
            result = (
                wavevector_from_energy(
                    energy_e, self.cmat.m_e, potential_offset=self.ue
                ),
                wavevector_from_energy(energy_e, self.smat.m_e),
            )
        return result

    def calculate_s1_energies(self, bounds=(), resolution=1000):
        lower_bound_e = self.ue + 1e-14
        upper_bound_e = 10 * self.ue
        lower_bound_h = self.uh + 1e-14
        upper_bound_h = 10 * self.uh

        x = np.linspace(lower_bound_e, upper_bound_e, resolution)
        if bounds != ():
            x = np.linspace(bounds[0], bounds[1], resolution)
        ye = electron_eigenvalue_residual(x, self)
        # print(np.all(np.isreal(yh)))

        ye_signs = np.sign(ye)
        ye_sign_change = np.diff(ye_signs)  # This array is one element shorter.
        ye_neg2pos_change = np.argwhere(np.where(ye_sign_change > 0.5, 1, 0))
        root_position = ye_neg2pos_change[0]
        # print(*x[root_position:root_position + 2])
        s1_electron_energy = brentq(
            electron_eigenvalue_residual,
            x[root_position],
            x[root_position + 1],
            args=(self),
        )

        x = np.linspace(lower_bound_h, upper_bound_h, resolution)
        if bounds != ():
            x = np.linspace(bounds[2], bounds[3], resolution)
        yh = hole_eigenvalue_residual(x, self)
        yh_signs = np.sign(yh)
        yh_sign_change = np.diff(yh_signs)  # This array is one element shorter.
        yh_neg2pos_change = np.argwhere(np.where(yh_sign_change > 0.5, 1, 0))
        root_position = yh_neg2pos_change[0]
        # print(yh[root_position], yh[root_position + 1])
        # print(hole_eigenvalue_residual(x[root_position] - 1, self), hole_eigenvalue_residual(x[root_position + 1], self))
        s1_hole_energy = brentq(
            hole_eigenvalue_residual,
            x[root_position],
            x[root_position + 1],
            args=(self),
        )
        # print(s1_electron_energy)
        # plt.plot(x, yh)
        # plt.ylim(-10, 10)
        # plt.show()
        return s1_electron_energy, s1_hole_energy

    def plot_electron_wavefunction(
        self, x, core_wavevector: float, shell_wavevector: float
    ):

        # x = np.linspace(1e-10, self.core_width + self.shell_width, 1000)
        cwf = lambda x: core_wavefunction(x, core_wavevector, self.core_width)
        swf = lambda x: shell_wavefunction(
            x, core_wavevector, self.core_width, self.shell_width
        )

        y = np.piecewise(
            x,
            [
                x < self.core_width,
                x > self.core_width,
                x > self.core_width + self.shell_width,
            ],
            [cwf, swf, 0],
        )
        return y

    def plot_potential_profile(self):
        """Plots one half of the spherically symmetric potential well of the quantum dot."""
        plt.hlines([self.cmat.vbe, self.cmat.cbe], xmin=0, xmax=self.core_width)
        plt.hlines(
            [self.smat.vbe, self.smat.cbe],
            xmin=self.core_width,
            xmax=self.core_width + self.shell_width,
        )
        lcbe, hcbe = sorted([self.cmat.cbe, self.smat.cbe])
        lvbe, hvbe = sorted([self.cmat.vbe, self.smat.vbe])
        plt.vlines(self.core_width, ymin=lcbe, ymax=hcbe)
        plt.vlines(self.core_width, ymin=lvbe, ymax=hvbe)
        # plt.vlines()
        plt.show()

    def analytical_overlap_integral(self):
        raise NotImplementedError
