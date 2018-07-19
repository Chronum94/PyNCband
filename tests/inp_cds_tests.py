import numpy as np
from scipy.constants import e

from pyncband import *


# @profile
def main():

    inp_effective_electron_mass = 0.079
    InP = Material(1.34, 0, inp_effective_electron_mass, 0.64, 9.6, "InP")
    CdS = Material(2.20, -0.39, 0.21, 0.68, 5.3, "CdS")

    shell_widths = [0.53, 1.05, 1.47, 1.90, 2.76, 3.84]

    experimental_bandgaps = [1.78, 1.46, 1.37, 1.32, 1.26, 1.24]
    print(
        "Using InP electron effective mass: {:0.2f}".format(inp_effective_electron_mass)
    )
    print("Core \t Shell \t Exp \t Whole \t\t Sectioned:")
    for i, shell_width in enumerate(shell_widths):
        csnc = CoreShellParticle(InP, CdS, 1.23, shell_width, 1.5)
        print(1.23, "\t", shell_width, end="\t")
        # print("Is CSNC type two? h/e?", csnc.type_two, csnc.h_e)
        energies = np.array(csnc.calculate_s1_energies())
        # print(energies)
        col_energy_whole, col_energy_sectioned = csnc.coulomb_screening_energy()
        pol_energy_whole, pol_energy_sectioned = csnc.interface_polarization_energy()
        whole_integral_energy = (
            csnc.bandgap + np.sum(energies) + col_energy_whole[0] + pol_energy_whole[0]
        )
        sectioned_integral_energy = (
            csnc.bandgap
            + np.sum(energies)
            + col_energy_sectioned[0]
            + pol_energy_sectioned[0]
        )
        # print("Col:", col_energy_whole, col_energy_sectioned, "Pol:", pol_energy)
        # print("NC bandgap:", csnc.bandgap)
        print(experimental_bandgaps[i], end="\t")
        print(
            "{:0.2f}({:0.2f})".format(
                whole_integral_energy,
                abs(experimental_bandgaps[i] - whole_integral_energy),
            ),
            end="\t",
        )
        print(
            "{:0.2f} ({:0.2f})".format(
                sectioned_integral_energy,
                abs(experimental_bandgaps[i] - sectioned_integral_energy),
            ),
            end="\t",
        )
        print()
        # print(csnc.localization_electron_shell(shell_width))
        # print(csnc.localization_hole_core())


if __name__ == "__main__":
    main()
