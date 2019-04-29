"""This module implements a CoreShellParticle class. It contains the necessary methods to carry out the first-order
order physics that we consider.

"""
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq

from .Material import Material
from .physicsfunctions import *
from .scaling import hbar_ev, m_e, wavenumber_nm_from_energy_ev
from .utils import EnergyNotBracketedError, LocalizationNotPossibleError

__all__ = ["CoreShellParticle"]


class CoreShellParticle:
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

        # I'll see if we need to scale these here. If all our calculations use scaled lengths,
        # we can simply work with nanometers.

        self.type_one, self.type_one_reverse = self._is_type_one()

        # Need to refactor this method/
        self.type_two, self.h_e, self.e_h = self._is_type_two()
        # This is an observer variable so we don't have to recalculate eigen-energies every time
        # unless core/shell dimensions change.
        self.energies_valid: bool = False
        self.s1_e, self.s1_h = None, None

        # Observer variable for normalization constant of wavefunction.
        self.norm_valid: bool = False
        self.norm_e, self.norm_h = None, None

        # All these energies are in eV. All internal calculations are carried out in eV.
        self.ue = np.abs(self.cmat.cbe - self.smat.cbe)
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe)

        self.bandgap = min(self.cmat.cbe, self.smat.cbe) - max(
            self.cmat.vbe, self.smat.vbe
        )

        self.environment_epsilon = environment_epsilon

        # This is used to refine the scanning of energies.
        # For coreshells with massive disparities, the energy scan_and_bracket needs 'adaptive' refinement.

        # TODO: Remove this/turn this into a runtime method.
        self.scan_refinement_multiplier = max(core_width, shell_width) / min(
            core_width, shell_width
        )

        self.BASE_ENERGY_SCAN_RESOLUTION = 500
        self.MAX_ENERGY_BRACKETING_ATTEMPTS = 500
        self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV = 0.1
        self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV = 0.1

    def calculate_s1_energies(self, resolution: int = None) -> np.ndarray:
        """Calculates eigenenergies of the S1 exciton state in eV.

        This may be changed in future versions to calculate higher energy states.

        Parameters
        ----------

        resolution : int
            The number of points to use when scanning and bracketing.

        Returns
        -------
        s1_energies : 2-array of float, eV
            The s1 state energies of electrons and holes.

        """

        # This is a heuristic way to scan finer points for larger quantum dots since their energies are lower.
        if resolution is None:
            resolution = int(
                self.BASE_ENERGY_SCAN_RESOLUTION * self.scan_refinement_multiplier
            )

        # _e for electrons, _h for holes.
        lower_bound_e = 0
        upper_bound_e = self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV
        lower_bound_h = 0
        upper_bound_h = self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV

        # Energy brackets.
        electron_bracket_found, hole_bracket_found = False, False
        current_electron_bracketing_attempt, current_hole_bracketing_attempt = 0, 0

        # Electron eigenvalue residual, defined here so we don't have to pass in 'self' in the rootfinder.
        # Might be worth testing for speed/performance.
        # TODO: Get rid of this intermediate currying and alter scan_and_bracket accordingly for all instances.
        def eer(x):
            return electron_eigenvalue_residual(x, self)

        while (
                not electron_bracket_found
                and current_electron_bracketing_attempt
                <= self.MAX_ENERGY_BRACKETING_ATTEMPTS
        ):
            (bracket_low, bracket_high), electron_bracket_found = scan_and_bracket(
                eer, lower_bound_e, upper_bound_e, resolution
            )
            lower_bound_e += self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV
            upper_bound_e += self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV
            current_electron_bracketing_attempt += 1

        if not electron_bracket_found:
            raise EnergyNotBracketedError(
                f"Energy was not bracketed after {self.MAX_ENERGY_BRACKETING_ATTEMPTS} scans "
                f"increasing by {self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV} eV each. Consider "
                "increaseing MAX_ENERGY_BRACKETING_ATTEMPTS or "
                "DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV, or both."
            )
        # Use the bracket to find the root.
        self.s1_e = brentq(eer, bracket_low, bracket_high)

        # Hole eigenvalue residual.
        def her(x):
            return hole_eigenvalue_residual(x, self)

        while (
                not hole_bracket_found
                and current_hole_bracketing_attempt <= self.MAX_ENERGY_BRACKETING_ATTEMPTS
        ):
            (bracket_low, bracket_high), hole_bracket_found = scan_and_bracket(
                her, lower_bound_h, upper_bound_h, resolution
            )
            lower_bound_h += self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV
            upper_bound_h += self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV
            current_hole_bracketing_attempt += 1

        if not hole_bracket_found:
            raise EnergyNotBracketedError(
                f"Energy was not bracketed after {self.MAX_ENERGY_BRACKETING_ATTEMPTS} scans "
                f"increasing by {self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV} eV each. Consider "
                "increaseing MAX_ENERGY_BRACKETING_ATTEMPTS or "
                "DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV, or both."
            )

        # Use bracket, find root.
        self.s1_h = brentq(her, bracket_low, bracket_high)

        self.energies_valid = True

        return np.array([self.s1_e, self.s1_h])