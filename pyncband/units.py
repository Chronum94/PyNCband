import pint as p


__all__ = ["emass", "eV", "nm", "Bohr", "Hartree"]
ur = p.UnitRegistry()

emass = ur.electron_mass

eV = ur.eV
Hartree = 27.21138602 * ur.eV

nm = ur.nm
Bohr = 0.0529177249 * ur.nm
# print(Hartree)
