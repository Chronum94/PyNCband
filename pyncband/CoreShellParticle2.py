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

        self.core_electron_potential_offset = 0 if core_material.cbe < shell_material.cbe else (core_material.cbe - shell_material.cbe)
        self.shell_electron_potential_offset = 0 if shell_material.cbe < core_material.cbe else (shell_material.cbe - core_material.cbe)

        self.core_hole_potential_offset = 0 if core_material.vbe > shell_material.vbe else -(
                    core_material.vbe - shell_material.vbe)
        self.shell_hole_potential_offset = 0 if shell_material.vbe > core_material.vbe else -(
                    shell_material.vbe - core_material.vbe)

        # I'll see if we need to scale these here. If all our calculations use scaled lengths,
        # we can simply work with nanometers.

        # This is an observer variable so we don't have to recalculate eigen-energies every time
        # unless core/shell dimensions change.


        # Observer variable for normalization constant of wavefunction.
        self.norm_e, self.norm_h = None, None

        # All these energies are in eV. All internal calculations are carried out in eV.
        self.ue = np.abs(self.cmat.cbe - self.smat.cbe)
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe)

        self.bandgap = min(self.cmat.cbe, self.smat.cbe) - max(
            self.cmat.vbe, self.smat.vbe
        )

        self.environment_epsilon = environment_epsilon


    def calculator_energy(
        self,
        core_width: float,
        shell_width: float,
        coulomb_interaction: bool = True,
        polarization_interaction: bool = True,
    ):
        raise NotImplementedError

    def calculate_s1_energies(self, core_width: float, shell_width: float, resolution: int = None) -> np.ndarray:
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
