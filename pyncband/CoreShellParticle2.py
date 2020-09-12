"""This module implements a CoreShellParticle class. It contains the necessary methods to carry out the first-order
order physics that we consider.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq

from .Material import Material
from .physicsfunctions2 import (
    _wavefunction,
    k2e,
    e2k,
    eigenvalue_residual,
    _core_wavefunction,
    _shell_wavefunction,
    _densityfunction
)

from typing import Union
# from .constants import ev_to_hartree, hartree_to_ev


__all__ = ["CoreShellParticle2"]


class CoreShellParticle2:
    def __init__(
        self,
        core_material: Material,
        shell_material: Material,
        environment_epsilon: float = 1.0,
    ):
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
        assert isinstance(core_material, Material)
        assert isinstance(shell_material, Material)

        self.name = core_material.name + "/" + shell_material.name

        self.cmat = core_material
        self.smat = shell_material

        """We reference all energy levels at 0, and all energy steps are positive wrt that level.
        Example: If the core material's conduction band is below (more negative) than the shell material's,
        the core material's conduction band edge is the reference energy set to 0. Switch signs to get hole energy
        alignments."""

        # All these energies are in Hartree (or decimal factors).
        # All internal calculations are carried out in Hartree (or decimal factors).
        # print(np.abs(self.cmat.vbe - self.smat.vbe))
        self.ue = np.abs(self.cmat.cbe - self.smat.cbe)
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe)

        self.core_electron_potential_offset = 0.0 if core_material.cbe < shell_material.cbe else self.ue
        self.shell_electron_potential_offset = 0.0 if shell_material.cbe < core_material.cbe else self.ue

        self.core_hole_potential_offset = 0.0 if core_material.vbe > shell_material.vbe else self.uh
        self.shell_hole_potential_offset = 0.0 if shell_material.vbe > core_material.vbe else self.uh

        # Spatially indirect bandgap.
        self.band_gap = np.min([self.cmat.cbe, self.smat.cbe]) - np.max([self.cmat.vbe, self.smat.vbe])
        # print(self.bandgap)
        self.environment_epsilon = environment_epsilon

        self.electron_wfn_norm = None
        self.hole_wfn_norm = None

    def calculator_energy(
        self,
        core_radius: float,
        shell_thickness: float,
        coulomb_interaction: bool = True,
        polarization_interaction: bool = True,
        output_all_energies: bool = False,
    ):
        coulomb_energy = 0.0
        polarization_energy = 0.0
        electron_s1_energy, hole_s1_energy = self._calculate_s1_energies(core_radius, shell_thickness)

        electron_core_wavenumber = e2k(electron_s1_energy, self.cmat.m_e, self.core_electron_potential_offset)
        electron_shell_wavenumber = e2k(electron_s1_energy, self.smat.m_e, self.shell_electron_potential_offset)

        hole_core_wavenumber = e2k(hole_s1_energy, self.cmat.m_h, self.core_hole_potential_offset)
        hole_shell_wavenumber = e2k(hole_s1_energy, self.smat.m_h, self.shell_hole_potential_offset)

        self.electron_wfn_norm = self._calculate_normalization(
            core_radius,
            shell_thickness,
            electron_core_wavenumber,
            electron_shell_wavenumber,
        )
        self.hole_wfn_norm = self._calculate_normalization(
            core_radius, shell_thickness, hole_core_wavenumber, hole_shell_wavenumber
        )

        if coulomb_interaction:
            coulomb_energy = self._calculate_coulomb_energy(core_radius, shell_thickness)
        if polarization_interaction:
            polarization_energy = self._calculate_polarization_energy(core_radius, shell_thickness)

        if output_all_energies:
            return (
                electron_s1_energy,
                hole_s1_energy,
                coulomb_energy,
                polarization_energy,
            )
        else:
            return electron_s1_energy + hole_s1_energy + coulomb_energy + polarization_energy

    def _calculate_s1_energies(self, core_radius: float, shell_thickness: float) -> np.ndarray:
        """Calculates eigenenergies of the S1 exciton state in eV.

        Parameters
        ----------
        core_radius: float, Bohr

        shell_thickness: float, Bohr

        Returns
        -------
        s1_energies : 2-array of float, Hartree
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

        electron_s1_energy = brentq(
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

        hole_s1_energy = brentq(
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

        return np.array([electron_s1_energy, hole_s1_energy])

    def _calculate_overlap_integral(self, core_radius, shell_thickness):
        electron_s1_energy, hole_s1_energy = self._calculate_s1_energies(core_radius, shell_thickness)
        k_electron, k_hole = (
            e2k(electron_s1_energy, self.cmat.m_e, self.core_electron_potential_offset),
            e2k(hole_s1_energy, self.cmat.m_h, self.core_hole_potential_offset),
        )
        q_electron, q_hole = (
            e2k(electron_s1_energy, self.smat.m_e, self.shell_electron_potential_offset),
            e2k(hole_s1_energy, self.smat.m_h, self.shell_hole_potential_offset),
        )
        # The integrals are done in 2 parts due to the possibility of sharp kinks at the interfaces.
        electron_wavefunction_radial_integral = quad(
            _core_wavefunction, 0, core_radius, args=(k_electron, core_radius)
        ) + quad(
            _core_wavefunction,
            core_radius,
            shell_thickness,
            args=(q_electron, shell_thickness),
        )

        hole_wavefunction_radial_integral = quad(_core_wavefunction, 0, core_radius, args=(k_hole, core_radius)) + quad(
            _core_wavefunction,
            core_radius,
            shell_thickness,
            args=(q_hole, shell_thickness),
        )

        # return electron_wavefunction_radial_integral[0], hole_wavefunction_radial_integral[0]
        #
        def core_overlap(r):
            return r * r * _core_wavefunction(r, k_electron, core_radius) * _core_wavefunction(r, k_hole, core_radius)

        def shell_overlap(r):
            return (
                r
                * r
                * _shell_wavefunction(r, q_electron, core_radius, shell_thickness)
                * _shell_wavefunction(r, q_hole, core_radius, shell_thickness)
            )

        overlap = quad(core_overlap, 0, core_radius) + quad(shell_overlap, core_radius, shell_thickness)
        return overlap

    def _plot_wf(self, electron_s1, hole_s1, core_radius, shell_thickness):
        k_electron, k_hole = (
            e2k(electron_s1, self.cmat.m_e, self.core_electron_potential_offset),
            e2k(hole_s1, self.cmat.m_h, self.core_hole_potential_offset),
        )
        q_electron, q_hole = (
            e2k(electron_s1, self.smat.m_e, self.shell_electron_potential_offset),
            e2k(hole_s1, self.smat.m_h, self.shell_hole_potential_offset),
        )

        x, dx = np.linspace(1e-8, core_radius + shell_thickness - 1e-8, 1000, retstep=True)
        electron_wavefunction = np.zeros_like(x)
        hole_wavefunction = np.zeros_like(x)
        electron_mass_weighted_wavefunction_derivative = np.zeros_like(x)
        hole_mass_weighted_wavefunction_derivative = np.zeros_like(x)

        # Array indices of core and shell regions.
        core_x, shell_x = (
            x < core_radius,
            np.logical_and(core_radius <= x, x < core_radius + shell_thickness),
        )

        # Building the electron and hole radial wavefunctions.
        electron_wavefunction[core_x] = (
            np.sin(x[core_x] * k_electron) / (x[core_x] * np.sin(k_electron * core_radius))
        ).real
        electron_wavefunction[shell_x] = (
            np.sin((core_radius + shell_thickness - x[shell_x]) * q_electron)
            / (x[shell_x] * np.sin(q_electron * shell_thickness))
        ).real
        electron_wavefunction_normalization_constant = 1 / (4 * np.pi * np.sum(electron_wavefunction) * dx)
        electron_wavefunction *= electron_wavefunction_normalization_constant
        # TODO: Shouldn't the abs-squared be normed?

        hole_wavefunction[core_x] = (np.sin(x[core_x] * k_hole) / (x[core_x] * np.sin(k_hole * core_radius))).real
        hole_wavefunction[shell_x] = (
            np.sin((core_radius + shell_thickness - x[shell_x]) * q_hole)
            / (x[shell_x] * np.sin(q_hole * shell_thickness))
        ).real
        hole_wavefunction_normalization_constant = 1 / (4 * np.pi * np.sum(hole_wavefunction) * dx)
        hole_wavefunction *= hole_wavefunction_normalization_constant

        # A uniform grid of masses for the mass-weighted derivatives.
        electron_mass_grid = np.where(x < core_radius, self.cmat.m_e, self.smat.m_e)
        hole_mass_grid = np.where(x < core_radius, self.cmat.m_h, self.smat.m_h)
        # print(np.sum(electron_wavefunction) * dx)
        # print(np.sum(hole_wavefunction) * dx)
        assert np.isclose(
            4 * np.pi * np.sum(electron_wavefunction) * dx, 1
        ), "1S electron wavefunction is not normalized! This should NOT happen here!"
        assert np.isclose(
            4 * np.pi * np.sum(hole_wavefunction) * dx, 1
        ), "1S hole wavefunction is not normalized! This should NOT happen here!"

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("Wavefunction")
        ax[1].set_title("Mass-weighted wavefunction derivative")

        ax[0].plot(x, electron_wavefunction)
        ax[0].plot(x, hole_wavefunction)

        electron_mass_weighted_wavefunction_derivative[core_x] = k_electron * np.cos(k_electron * x[core_x]) / (
            np.sin(k_electron * core_radius) * x[core_x]
        ) - np.sin(k_electron * x[core_x]) / (np.sin(core_radius * k_electron) * x[core_x] ** 2)
        electron_mass_weighted_wavefunction_derivative[shell_x] = -q_electron * np.cos(
            q_electron * (-x[shell_x] + core_radius + shell_thickness)
        ) / (np.sin(q_electron * shell_thickness) * x[shell_x]) - np.sin(
            q_electron * (-x[shell_x] + core_radius + shell_thickness)
        ) / (
            np.sin(shell_thickness * q_electron) * x[shell_x] ** 2
        )

        electron_mass_weighted_wavefunction_derivative = electron_mass_weighted_wavefunction_derivative.real

        hole_mass_weighted_wavefunction_derivative[core_x] = k_hole * np.cos(k_hole * x[core_x]) / (
            np.sin(k_hole * core_radius) * x[core_x]
        ) - np.sin(k_hole * x[core_x]) / (np.sin(core_radius * k_hole) * x[core_x] ** 2)
        hole_mass_weighted_wavefunction_derivative[shell_x] = -q_hole * np.cos(
            q_hole * (-x[shell_x] + core_radius + shell_thickness)
        ) / (np.sin(q_hole * shell_thickness) * x[shell_x]) - np.sin(
            q_hole * (-x[shell_x] + core_radius + shell_thickness)
        ) / (
            np.sin(shell_thickness * q_hole) * x[shell_x] ** 2
        )

        hole_mass_weighted_wavefunction_derivative = hole_mass_weighted_wavefunction_derivative.real

        ax[1].plot(
            x,
            electron_mass_weighted_wavefunction_derivative
            * electron_wavefunction_normalization_constant
            / electron_mass_grid,
        )
        # ax[1].plot(x, np.gradient(electron_wavefunction, dx) / electron_mass_grid, "C0-")
        ax[1].plot(
            x,
            hole_mass_weighted_wavefunction_derivative * hole_wavefunction_normalization_constant / hole_mass_grid,
        )
        # ax[1].plot(x, np.gradient(hole_wavefunction, dx) / hole_mass_grid, "C0-")

        # plt.vlines([core_radius, core_radius + shell_thickness], 0, np.max(electron_wavefunction.real))
        # plt.xlim(core_radius - 3 * dx, core_radius + 3 * dx)
        plt.show()

    def _calculate_normalization(self, core_radius: float, shell_thickness: float, core_wavenumber: Union[float, complex], shell_wavenumber: Union[float, complex]):
        density_integral = quad(
            lambda x: x * x * _densityfunction(x, core_wavenumber, shell_wavenumber, core_radius, shell_thickness),
            0,
            core_radius + shell_thickness,
        )[0]

        return 1 / density_integral

    def _calculate_coulomb_energy(self, core_radius, shell_thickness):
    def coulomb_screening_energy(self, relative_tolerance: float = 1e-5, shell_term_denominator: float = 2.0):
        """Calculates the Coulomb screening energy. Somewhat slow.

        Parameters
        ----------
        relative_tolerance : float
            The relative tolerance for the Coulomb screening energy integral. Defaults to 1e-5.

        plot_integrand : bool

        cmap : str

        shell_term_denominator : float
            Equation 8 in Piryatinski/Klimov NanoLett '07 has a factor of two in the second term's denominator.
            However, the model matches up to experiment closer when this term is 1. For correctness, I've kept it
            default to 0, but it is possible that there might be wrong (from the original literature) somewhere.

        Returns
        -------
        2-array of floats: The Coulomb screening energy and error.

        """
        coulomb_screening_operator = make_coulomb_screening_operator(
            self, shell_term_denominator=shell_term_denominator
        )
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
        norm_e, norm_h = self._normalization()

        # Electron/hole density functions.
        def edf(x):
            return _densityfunction(x, k_e, q_e, self.core_width, self.shell_width)

        def hdf(x):
            return _densityfunction(x, k_h, q_h, self.core_width, self.shell_width)

        coulomb_integrand = lambda r1, r2: r1 ** 2 * r2 ** 2 * edf(r1) * hdf(r2) * coulomb_screening_operator(r1,
                                                                                                              r2)

        # Energy returned in units of eV.
        # r1 < R, r2 < R
        region_one = np.array(
            dblquad(
                coulomb_integrand,
                0,
                self.core_width,
                0,
                self.core_width,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        # r1 > R, r2 < R

        region_two = np.array(
            dblquad(
                coulomb_integrand,
                self.core_width,
                self.radius,
                0,
                self.core_width,
                epsrel=relative_tolerance,
            )
        )

        # r1 > R, r2 > R
        region_three = np.array(
            dblquad(
                coulomb_integrand,
                self.core_width,
                self.radius,
                self.core_width,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        # r1 < R, r2 > R
        region_four = np.array(
            dblquad(
                coulomb_integrand,
                0,
                self.core_width,
                self.core_width,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )
        sectioned_integral = (region_one + region_two + region_three + region_four) * norm_h * norm_e
    pass


def get_type(self, detailed: bool = False) -> str:
    return "Soon..."
