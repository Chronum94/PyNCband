"""This file contains a simple material class, containing only the properties of the material that we need."""

__all__ = ["Material"]


class Material:
    def __init__(
        self,
        bandgap: float,
        conduction_band_edge: float,
        electron_eff_mass: float,
        hole_eff_mass: float,
        epsilon: float,
        name: str = None,
    ):
        """Create a simple material.

        A minimalist material properties class to hold the material properties that we need for the relevant
        calculations in this package.

        Parameters
        ----------

        bandgap : float, eV
            A real, positive value for the band gap of the semiconductor material.

        conduction_band_edge : float, eV
            A real, negative value for the conduction band edge with reference to the vacuum level of the material.

        electron_eff_mass : float, electron-mass
            The isotropic effective mass of the electron in the conduction band.

        hole_eff_mass : float, electron-mass
            The isotropic effective mass of the hole in the valence band.

        epsilon : float, free-space-permittivity
            The real dielectric constant of the material, in terms of the permittivity of free space.

        name : str
            The name of the material.

        """

        self.bandgap: float = bandgap

        self.cbe: float = conduction_band_edge
        self.vbe: float = conduction_band_edge - bandgap

        self.m_e: float = electron_eff_mass
        self.m_h: float = hole_eff_mass
        self.eps: float = epsilon
        self.name: str = name

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            return self.__repr__()
