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

        bandgap : float
            A real, positive value for the band gap of the semiconductor material.

        conduction_band_edge : float
            A real, negative value for the conduction band edge with reference to the vacuum level of the material.

        electron_eff_mass : float
            The isotropic effective mass of the electron in the conduction band.

        hole_eff_mass : float
            The isotropic effective mass of the hole in the valence band.

        epsilon : float
            The real dielectric constant of the material. Dielectric constants are usually complex, so not sure what's
            happening here.

        name : str
            The name of the material."""

        self.bandgap = bandgap

        self.cbe = conduction_band_edge
        self.vbe = conduction_band_edge - bandgap

        self.m_e = electron_eff_mass
        self.m_h = hole_eff_mass
        self.eps = epsilon
        self.name = name

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            return self.__repr__()
