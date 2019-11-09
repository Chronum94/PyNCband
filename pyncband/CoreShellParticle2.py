"""This module implements a CoreShellParticle class. It contains the necessary methods to carry out the first-order
order physics that we consider.

"""
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq

from .Material2 import Material2
from .physicsfunctions2 import *

# from .constants import ev_to_hartree, hartree_to_ev
from .units import Hartree, Bohr, eV, nm
from .utils import EnergyNotBracketedError, LocalizationNotPossibleError

__all__ = ["CoreShellParticle2"]


class CoreShellParticle2:
    def __init__(self, core_material: Material2, shell_material: Material2, environment_epsilon: float = 1.0):
        """Creates a core-shell nanoparticle.

        Parameters
        ----------
        core_material : Material
            A Material object representing the core material of the nanocrystal.

        shell_material : Material
            A Material object representing the shell material of the nanocrystal.

        environment_epsilon: float
            A float representing the relative permittivity of the medium in which the QD is.
        """
        # Preliminary asserts.
        assert isinstance(core_material, Material2)
        assert isinstance(shell_material, Material2)

        self.name = core_material.name + "/" + shell_material.name

        self.cmat = core_material
        self.smat = shell_material

        """We reference all energy levels at 0, and all energy steps are positive wrt that level.
        Example: If the core material's conduction band is below (more negative) than the shell material's,
        the core material's condcution band edge is the reference energy set to 0. Switch signs to get hole energy
        alignments."""

        # All these energies are in Hartree. All internal calculations are carried out in Hartree.
        # print(np.abs(self.cmat.vbe - self.smat.vbe))
        self.ue = np.abs(self.cmat.cbe - self.smat.cbe)
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe)

        self.core_electron_potential_offset = 0.0 if core_material.cbe < shell_material.cbe else self.ue
        self.shell_electron_potential_offset = 0.0 if shell_material.cbe < core_material.cbe else self.ue

        self.core_hole_potential_offset = 0.0 if core_material.vbe > shell_material.vbe else self.uh
        self.shell_hole_potential_offset = 0.0 if shell_material.vbe > core_material.vbe else self.uh

        # Spatially indirect bandgap.
        self.bandgap = np.min([self.cmat.cbe, self.smat.cbe]) - np.max([self.cmat.vbe, self.smat.vbe])
        print(self.bandgap)
        self.environment_epsilon = environment_epsilon

    def calculator_energy(
        self,
        core_width: float,
        shell_width: float,
        coulomb_interaction: bool = True,
        polarization_interaction: bool = True,
    ):
        raise NotImplementedError

    def _calculate_s1_energies(self, core_radius: float, shell_thickness: float) -> np.ndarray:
        """Calculates eigenenergies of the S1 exciton state in eV.

        Parameters
        ----------
        core_radius: float, nanometers

        shell_thickness: float, nanometers

        Returns
        -------
        s1_energies : 2-array of float, eV
            The s1 state energies of electrons and holes.

        References
        ----------
        .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

        """

        # The size paramater at which the two x cot(x) terms diverge.
        k_bracket, q_bracket = np.pi / core_radius, np.pi / shell_thickness
        print("K bracket:", k_bracket, "Q bracket:", q_bracket)
        # print(self.core_electron_potential_offset, self.shell_electron_potential_offset)
        # print(self.core_hole_potential_offset, self.shell_hole_potential_offset)
        # These are the energies which act as the nonzero brackets of our roots.
        electron_bracket = min(
            k2e(k_bracket, self.cmat.m_e, self.core_electron_potential_offset),
            k2e(q_bracket, self.smat.m_e, self.shell_electron_potential_offset),
        )

        hole_bracket = min(
            k2e(k_bracket, self.cmat.m_h, self.core_hole_potential_offset),
            k2e(q_bracket, self.smat.m_h, self.shell_hole_potential_offset),
        )

        print("E bracket:", electron_bracket, "H bracket:", hole_bracket)

        electron_s1 = brentq(
            eigenvalue_residual,
            0.0,
            electron_bracket - 1e-9,
            args=(
                core_radius,
                shell_thickness,
                self.cmat.m_e,
                self.smat.m_e,
                self.core_electron_potential_offset,
                self.shell_electron_potential_offset,
            ),
        )

        hole_s1 = brentq(
            eigenvalue_residual,
            0.0,
            hole_bracket - 1e-9,
            args=(
                core_radius,
                shell_thickness,
                self.cmat.m_h,
                self.smat.m_h,
                self.core_hole_potential_offset,
                self.shell_hole_potential_offset,
            ),
        )

        return np.array([electron_s1, hole_s1])

    def _plot_wf(self, electron_s1, hole_s1, core_radius, shell_thickness):
        k_electron, k_hole = (
            e2k(electron_s1, self.cmat.m_e, self.core_electron_potential_offset),
            e2k(hole_s1, self.cmat.m_h, self.core_hole_potential_offset),
        )
        q_electron, q_hole = (
            e2k(electron_s1, self.smat.m_e, self.shell_electron_potential_offset),
            e2k(hole_s1, self.smat.m_h, self.shell_hole_potential_offset),
        )

        import matplotlib.pyplot as plt

        x, dx = np.linspace(1e-8, core_radius + shell_thickness - 1e-8, 1000, retstep=True)
        electron_wavefunction = np.zeros_like(x)
        hole_wavefunction = np.zeros_like(x)
        electron_mass_weighted_wavefunction_derivative = np.zeros_like(x)
        hole_mass_weighted_wavefunction_derivative = np.zeros_like(x)

        # Array indices of core and shell regions.
        core_x, shell_x = (x < core_radius, np.bitwise_and(core_radius <= x, x < core_radius + shell_thickness))

        # Building the electron and hole radial wavefunctions.
        electron_wavefunction[core_x] = (
            np.sin(x[core_x] * k_electron) / (x[core_x] * np.sin(k_electron * core_radius))
        ).real
        electron_wavefunction[shell_x] = (
            np.sin((core_radius + shell_thickness - x[shell_x]) * q_electron)
            / (x[shell_x] * np.sin(q_electron * shell_thickness))
        ).real
        electron_wavefunction_normalization_constant = 1 / (np.sum(electron_wavefunction) * dx)
        electron_wavefunction *= electron_wavefunction_normalization_constant

        hole_wavefunction[core_x] = (np.sin(x[core_x] * k_hole) / (x[core_x] * np.sin(k_hole * core_radius))).real
        hole_wavefunction[shell_x] = (
            np.sin((core_radius + shell_thickness - x[shell_x]) * q_hole)
            / (x[shell_x] * np.sin(q_hole * shell_thickness))
        ).real
        hole_wavefunction_normalization_constant = 1 / (np.sum(hole_wavefunction) * dx)
        hole_wavefunction *= hole_wavefunction_normalization_constant

        # A uniform grid of masses for the mass-weighted derivatives.
        electron_mass_grid = np.where(x < core_radius, self.cmat.m_e, self.smat.m_e)
        hole_mass_grid = np.where(x < core_radius, self.cmat.m_h, self.smat.m_h)
        print(np.sum(electron_wavefunction) * dx)
        print(np.sum(hole_wavefunction) * dx)
        assert np.isclose(np.sum(electron_wavefunction) * dx, 1), "Electron wavefunction is not normalized! This should NOT happen here!"
        assert np.isclose(np.sum(hole_wavefunction) * dx, 1), "Hole wavefunction is not normalized! This should NOT happen here!"

        fig, ax= plt.subplots(1, 2, figsize = (10, 5))
        ax[0].set_title('Wavefunction')
        ax[1].set_title('Mass-weighted wavefunction derivative')



        ax[0].plot(x, electron_wavefunction)
        ax[0].plot(x, hole_wavefunction)


        electron_mass_weighted_wavefunction_derivative[core_x] = k_electron * np.cos(k_electron * x[core_x]) / (np.sin(k_electron * core_radius) * x[core_x]) \
        - np.sin(k_electron * x[core_x]) / (np.sin(core_radius * k_electron) * x[core_x] ** 2)
        electron_mass_weighted_wavefunction_derivative[shell_x] = - q_electron * np.cos(q_electron * (- x[shell_x] + core_radius + shell_thickness)) / (
                    np.sin(q_electron * shell_thickness) * x[shell_x]) \
                                                                 - np.sin(q_electron * (- x[shell_x] + core_radius + shell_thickness)) / (
                                                                             np.sin(shell_thickness * q_electron) * x[
                                                                         shell_x] ** 2)

        electron_mass_weighted_wavefunction_derivative = electron_mass_weighted_wavefunction_derivative.real

        hole_mass_weighted_wavefunction_derivative[core_x] = k_hole * np.cos(k_hole * x[core_x]) / (
                    np.sin(k_hole * core_radius) * x[core_x]) \
                                                                 - np.sin(k_hole * x[core_x]) / (
                                                                             np.sin(core_radius * k_hole) * x[
                                                                         core_x] ** 2)
        hole_mass_weighted_wavefunction_derivative[shell_x] = - q_hole * np.cos(
            q_hole * (- x[shell_x] + core_radius + shell_thickness)) / (
                                                                          np.sin(q_hole * shell_thickness) * x[
                                                                      shell_x]) \
                                                                  - np.sin(
            q_hole * (- x[shell_x] + core_radius + shell_thickness)) / (
                                                                          np.sin(shell_thickness * q_hole) * x[
                                                                      shell_x] ** 2)

        hole_mass_weighted_wavefunction_derivative = hole_mass_weighted_wavefunction_derivative.real



        ax[1].plot(x, electron_mass_weighted_wavefunction_derivative * electron_wavefunction_normalization_constant / electron_mass_grid, "C1--", lw=4)
        ax[1].plot(x, np.gradient(electron_wavefunction, dx) / electron_mass_grid, "C0-")
        ax[1].plot(x,
                   hole_mass_weighted_wavefunction_derivative * hole_wavefunction_normalization_constant / hole_mass_grid,
                   "C1--", lw=4)
        ax[1].plot(x, np.gradient(hole_wavefunction, dx) / hole_mass_grid, "C0-")

        # plt.vlines([core_radius, core_radius + shell_thickness], 0, np.max(electron_wavefunction.real))
        # plt.xlim(core_radius - 3 * dx, core_radius + 3 * dx)
        plt.show()


def get_type(self, detailed: bool = False) -> str:
    return "Soon..."
