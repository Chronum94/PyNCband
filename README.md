# PyNCband

PyNCband (pronounced pink-band) is a software suite in its _very_ early stages of development to predict the bandgaps and photoluminescence energies of core-shell (for now) quantum dots.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## For users:
This is still in development. You'll need PyCharm for any hope of using it.

## For developers:
I've tried to make the code as self-explanatory and lucid as possible. Nevertheless, I will be adding a contribution readme soon.

## Features:

- [x] Energy calculation of S1 exciton state.
- [x] Coulomb screening energy calculation.
- [x] Polarization interaction energy calculation.
- [x] Internal units consistency.
- [x] Wavenumbers for Type 1 NQDs.
- [ ] Verification against experimental data.

## Minimal example:

```python
import numpy as np
from scipy.constants import e

from pyncband import *

# Declare a material with a bandgap of 1.34 eV, conduction band edge offset of 0,
# electron(hole) effective mass of 0.07(0.64), and dielectric constant of 9.6. Optionally,
# give it a name.
InP = Material(1.34, 0, 0.07, 0.64, 9.6, 'InP')

# The conduction band edge is different here. For a core-shell nanocrystal, only the
# relative offset matters.
CdS = Material(2.20, -0.39, 0.21, 0.68, 5.3, 'CdS')

# Create a CoreShellParticle with the two materials, the core and shell thicknesses in nm.
# Optionally, supply a permittivity for the environment of the nanocrystal, if it isn't 1.
csnc = CoreShellParticle(InP, CdS, 1.23, 3.84, 1.5)

# This is a type two, h/e structure. Both of these should be true.
print("Is CSNC type two? h/e?", csnc.type_two, csnc.h_e)

# Calculate energies, in eV
energies = np.array(csnc.calculate_s1_energies())

# Print them out, because why not.
# Should be [0.09009009 0.27527528]
print(energies)

# Calculate the Coulomb screening energy, this is in eV already.
# These methods return both the energy and the uncertainty in the calculation.
col_energy = csnc.coulomb_screening_energy()
# The polarization interaction energy, also in eV already.
pol_energy = csnc.interface_polarization_energy()

# Col: [-1.50293370e-03  5.56435081e-07] Pol: [-1.98701001e-03  4.79341560e-07]
print('Col:', col_energy, 'Pol:', pol_energy)

# The bandgap of the QD.
print("NC bandgap:", csnc.bandgap)
# The excitation energy of the 1S exciton.
print("Net 1S energy:", csnc.bandgap + np.sum(energies) + col_energy[0] + pol_energy[0])
```