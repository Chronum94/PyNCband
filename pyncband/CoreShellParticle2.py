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

        self.core_electron_potential_offset = (
            0.0
            if core_material.cbe < shell_material.cbe
            else (core_material.cbe - shell_material.cbe) * ev_to_hartree
        )
        self.shell_electron_potential_offset = (
            0.0
            if shell_material.cbe < core_material.cbe
            else (shell_material.cbe - core_material.cbe) * ev_to_hartree
        )

        self.core_hole_potential_offset = (
            0.0
            if core_material.vbe > shell_material.vbe
            else -(core_material.vbe - shell_material.vbe) * ev_to_hartree
        )
        self.shell_hole_potential_offset = (
            0.0
            if shell_material.vbe > core_material.vbe
            else -(shell_material.vbe - core_material.vbe) * ev_to_hartree
        )

        # I'll see if we need to scale these here. If all our calculations use scaled lengths,
        # we can simply work with nanometers.

        # All these energies are in Hartrees. All internal calculations are carried out in Hartrees.
        self.ue = np.abs(self.cmat.cbe - self.smat.cbe) * ev_to_hartree
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe) * ev_to_hartree

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

        This may be changed in future versions to calculate higher energy states.

        The s1 state energies are the roots of equation 2 in Ref 1.
        To solve this equation, take the RHS to the LHS
        to represent as f(k(e), q(e)) = 0 then
        1. replace kR=X and qH=Y.
        2. Notice that the equation now goes from negative to positive
        (plot it as a function of energy!) similar to a -x*cot(x), with some
        irregularities. At each asymptote, one of these two cots' arguments, X or Y,
        is hitting pi. That's why you have the x*cot(x) blowing up, and dominating
        the behaviour of the equation at those points. This behaviour is really
        useful, since the first asymptotic explosion will act as our upper bracket
        for the root-finding of the energy.

        2. If the state energy is less than the potential step at the core-shell interface
        then the electron is localized in either the core/shell, depending on which side
        has the potential step.
        The region where the electron wavefunction is evanescent will
        have a purely imaginary size parameter.
        An imaginary size parameter will not contribute to the asymptotic blow-up,
        as val*cot(val) goes to inf due to val -> pi^-, since cot(val) does not blow
        up for imaginary val.

        3. Find the wavenumber, and therefore the energy, where X (or Y, if
        the step is in the core) becomes pi.
        4. Find corresponding energy, given eff mass.
        5. Bracket [0, above result), find root.

        6. If the state energy is higher than the potenttial step, this means that X
        (or Y, if the step is in the core) has not hit pi for an energy lower than
        the potential step. Find the wavenumber, either k or q, such that X or Y hits pi.
        Whichever hits pi first, as above, will control the first asymptotic behaviour.
        We will use to bracket the first root, and solve.

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
