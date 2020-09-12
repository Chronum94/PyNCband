"""This file contains a simple material class/struct, containing only the properties of the material that we need.
Maybe a dataclass?"""

__all__ = ["Material"]
from typing import Union


class Material:
    def __init__(
        self,
        band_gap: float,
        electron_eff_mass: float,
        hole_eff_mass: float,
        epsilon: float,
        name: str = None,
        *,
        conduction_band_edge: Union[float, None],
        valence_band_edge: Union[float, None]
    ):
        """Create a simple material.

        A minimalist material properties class to hold the material properties that we need for the relevant
        calculations in this package.

        Parameters
        ----------

        band_gap : float
            A real, positive value for the band gap of the semiconductor material, in units of eV.

        electron_eff_mass : float
            The isotropic effective mass of the electron in the conduction band, in units of electron mass.

        hole_eff_mass : float, electron-mass
            The isotropic effective mass of the hole in the valence band, in units of electron mass.

        epsilon : float, free-space-permittivity
            The real dielectric constant of the material.

        name : str
            The name of the material.

        conduction_band_edge : float, eV
            A real, negative value for the conduction band edge with reference to the vacuum level of the material, in units of eV.

        valence_band_edge : float
            A real, negative value for the valence band edge with reference to the vacuum level of the material, in units of eV.

        """

        self.band_gap = band_gap

        assert not (conduction_band_edge and valence_band_edge)

        if conduction_band_edge is not None:
            self.cbe = conduction_band_edge
            self.vbe = conduction_band_edge - band_gap
        else:
            self.cbe = valence_band_edge + band_gap
            self.vbe = valence_band_edge

        self.m_e: float = electron_eff_mass
        self.m_h: float = hole_eff_mass
        self.eps: float = epsilon
        if name is not None:
            self.name: str = name
        else:
            self.name: str = "NO_NAME_GIVEN"

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            return self.__repr__()
