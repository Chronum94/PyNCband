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