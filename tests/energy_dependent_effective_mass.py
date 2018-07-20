import numpy as np

from pyncband import Material, CoreShellParticle


InP = Material(1.34, 0, 1.0, 0.64, 9.6, "InP")
CdS = Material(2.20, -0.39, 0.21, 0.68, 5.3, "CdS")


def inp_electron_mass_from_energy(e):
    alpha = -1.2
    e_p = 20.6
    e_g = 1.34
    return 1 / (alpha + e_p / (e_g + e))


for energy in np.linspace(0.1, 1, 5):
    print(inp_electron_mass_from_energy(energy))

inp_eff_electron_mass = 1.0
mass_eps = 1.0

print("Iteratively figuring out InP electron effective mass:")
while mass_eps > 1e-5:
    InP = Material(1.34, 0, inp_eff_electron_mass, 0.64, 9.6, "InP")
    csnc = CoreShellParticle(InP, CdS, 1.23, 1.90, 1.5)
    energy_e, energy_h = csnc.calculate_s1_energies()
    inp_eff_electron_mass_new = inp_electron_mass_from_energy(energy_e)
    mass_eps = abs(inp_eff_electron_mass_new - inp_eff_electron_mass)
    print(mass_eps, inp_eff_electron_mass, inp_eff_electron_mass_new)
    inp_eff_electron_mass = inp_eff_electron_mass_new
