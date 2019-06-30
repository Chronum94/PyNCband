"""This module implements a CoreShellParticle class. It contains the necessary methods to carry out the first-order
order physics that we consider.

"""
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq

from .Material import Material
from .physicsfunctions2 import *
from .constants import ev_to_hartree, hartree_to_ev
from .utils import EnergyNotBracketedError, LocalizationNotPossibleError

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
        Example: If the core material's conduction band is below (more negative) than the valence material,
        The core material's condcution band edge is the reference energy set to 0. Switch signs to get hole energy
        alignments."""

        # All these energies are in Hartrees. All internal calculations are carried out in Hartrees.
        self.ue = np.abs(self.cmat.cbe - self.smat.cbe) * ev_to_hartree
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe) * ev_to_hartree

        self.core_electron_potential_offset = (
            0.0
            if core_material.cbe < shell_material.cbe
            else self.ue
        )
        self.shell_electron_potential_offset = (
            0.0
            if shell_material.cbe < core_material.cbe
            else self.ue
        )

        self.core_hole_potential_offset = (
            0.0
            if core_material.vbe > shell_material.vbe
            else self.uh
        )
        self.shell_hole_potential_offset = (
            0.0
            if shell_material.vbe > core_material.vbe
            else self.uh
        )

        # Spatially indirect bandgap.
        self.bandgap = (
            min(self.cmat.cbe, self.smat.cbe) - max(self.cmat.vbe, self.smat.vbe)
        ) * ev_to_hartree

        self.environment_epsilon = environment_epsilon

    def calculator_energy(
        self,
        core_width: float,
        shell_width: float,
        coulomb_interaction: bool = True,
        polarization_interaction: bool = True,
    ):
        raise NotImplementedError

    def _calculate_s1_energies(
        self, core_radius: float, shell_thickness: float
    ) -> np.ndarray:
        """Calculates eigenenergies of the S1 exciton state in eV.

        Parameters
        ----------

        Returns
        -------
        s1_energies : 2-array of float, eV
            The s1 state energies of electrons and holes.

        """

        # The size paramater at which the two x cot(x) terms diverge.
        k_bracket, q_bracket = np.pi / core_radius, np.pi / shell_thickness
        print( "K bracket:", k_bracket, "Q bracket:", q_bracket)
        print(self.core_electron_potential_offset, self.shell_electron_potential_offset * hartree_to_ev)
        print(self.core_hole_potential_offset, self.shell_hole_potential_offset * hartree_to_ev)
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


def get_type(self, detailed: bool = False) -> str:
    return "Soon..."
