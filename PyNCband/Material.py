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
        """Create a simple material with a positive bandgap in eV, and a negative conduction band edge in eV.
        The epsilon is necessary for Coulomb interaction calculations."""
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
