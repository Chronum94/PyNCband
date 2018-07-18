from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar, e, m_e
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq

from .Material import Material
from .physicsfunctions import *
from .scaling import n_
from .utils import *

__all__ = ["CoreShellParticle"]


class CoreShellParticle:
    def __init__(
        self,
        core_material: Material,
        shell_material: Material,
        core_width: float,
        shell_width: float,
        environment_epsilon: float = 1.0
    ):
        """Creates a core-shell nanoparticle.

        Parameters
        ----------
        core_material : Material
        shell_material : Material
        core_width : float, nanometers
        shell_width : float, nanometers
        """

        self.cmat = core_material
        self.smat = shell_material

        # I'll see if we need to scale these here. If all our calculations use scaled lengths,
        # we can simply work with nanometers.

        self.core_width = core_width
        self.shell_width = shell_width
        self.radius = core_width + shell_width
        self.type_one = self._is_type_one()

        # Need to refactor this method/
        self.type_two, self.h_e, self.e_h = self._is_type_two()
        # This is an observer variable so we don't have to recalculate eigen-energies every time.
        self.energies_valid: bool = False
        self.s1_e, self.s1_h = None, None

        # Observer variable for normalization constant of wavefunction.
        self.norm_valid: bool = False
        self.norm_e, self.norm_h = None, None

        # Band alignment energies, in Joules. I should see if we actually need them in Joules or not.
        self.ue = np.abs(self.cmat.cbe - self.smat.cbe) * e  # Converting to Joules.
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe) * e

        self.bandgap = min(self.cmat.cbe, self.smat.cbe) - max(
            self.cmat.vbe, self.smat.vbe
        )

        self.environment_epsilon = environment_epsilon

    def set_core_width(self, x):
        """

        Parameters
        ----------
        x : float, nanometers

        Returns
        -------

        """
        self.radius -= self.core_width
        self.core_width = x
        self.radius += self.core_width
        self.energies_valid = False
        self.norm_valid = False

    def set_shell_width(self, x):
        """

        Parameters
        ----------
        x : float, nanometers

        Returns
        -------

        """
        self.radius -= self.shell_width
        self.shell_width = x
        self.radius += self.shell_width
        self.energies_valid = False
        self.norm_valid = False

    def calculate_wavenumbers(self) -> np.ndarray:
        """Returns a tuple of the electron wavevectors in the core and the shell.

        Returns
        -------
        wavenumbers : [float, float, float, float] : Wavenumbers in 1 / m.
            Array of wavenumbers

        References
        ----------
        .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

        .. [2] Li, L., Reiss, P., & Protie, M. (2009). Core / Shell Semiconductor Nanocrystals, (2), 154–168. \
        https://doi.org/10.1002/smll.200800841

        """
        # energy_e, energy_h = None, None

        energy_e, energy_h = self.calculate_s1_energies()
        # print('E:', energy_e, energy_h)
        # This gets set to false when we change core/shell radius, etc.
        if self.type_one:
            return np.array(
                [
                    wavenumber_from_energy(
                        energy_e, self.cmat.m_e, potential_offset=self.ue
                    ),
                    wavenumber_from_energy(energy_e, self.smat.m_e),
                    wavenumber_from_energy(
                        energy_h, self.cmat.m_h, potential_offset=self.uh
                    ),
                    wavenumber_from_energy(energy_h, self.smat.m_h),
                ]
            )
        elif self.type_two:
            if self.e_h:
                return np.array(
                    [
                        wavenumber_from_energy(energy_e, self.cmat.m_e),
                        wavenumber_from_energy(
                            energy_e, self.smat.m_e, potential_offset=self.ue
                        ),
                        wavenumber_from_energy(
                            energy_h, self.cmat.m_h, potential_offset=self.uh
                        ),
                        wavenumber_from_energy(energy_h, self.smat.m_h),
                    ]
                )
            elif self.h_e:
                return np.array(
                    [
                        wavenumber_from_energy(
                            energy_e, self.cmat.m_e, potential_offset=self.ue
                        ),
                        wavenumber_from_energy(energy_e, self.smat.m_e),
                        wavenumber_from_energy(energy_h, self.cmat.m_h),
                        wavenumber_from_energy(
                            energy_h, self.smat.m_h, potential_offset=self.uh
                        ),
                    ]
                )

    # This method can currently only find cases where the energy of the lowest state is above the potential step.
    def calculate_s1_energies(self, bounds=(), resolution=1000) -> Tuple[float, float]:
        """Calculates eigenenergies of the S1 exciton state in Joules.

        Parameters
        ----------
        bounds : tuple of floats (e_lower, e_higher, h_lower, h_higher)
        resolution : int
            The number of points to use when scanning and bracketing.

        Returns
        -------
        s1_energies : 2-tuple of float, Joules
            The s1 exciton energies of electrons and holes.
        """

        # Bounds in Joules.
        # TODO: Find a better way to bracket energies.
        lower_bound_e = 0
        upper_bound_e = 5 * e
        lower_bound_h = 0
        upper_bound_h = 5 * e

        if bounds != ():
            upper_bound_e = bounds[0] * e

        # Electron eigenvalue residual.
        def eer(x):
            return electron_eigenvalue_residual(x, self)

        bracket = scan_and_bracket(eer, lower_bound_e, upper_bound_e, resolution)
        self.s1_e = brentq(electron_eigenvalue_residual, *bracket, args=(self,))

        if bounds != ():
            upper_bound_h = bounds[1] * e

        # Hole eigenvalue residual.
        def her(x):
            return hole_eigenvalue_residual(x, self)

        bracket = scan_and_bracket(her, lower_bound_h, upper_bound_h, resolution)

        self.s1_h = brentq(hole_eigenvalue_residual, *bracket, args=(self,))
        self.energies_valid = True

        return self.s1_e, self.s1_h

    def plot_electron_wavefunction(
        self, x, core_wavenumber: float, shell_wavenumber: float
    ):

        y = wavefunction(
            x, core_wavenumber, shell_wavenumber, self.core_width, self.shell_width
        )
        return y

    def plot_potential_profile(self):
        """Plots one half of the spherically symmetric potential well of the quantum dot."""
        plt.hlines([self.cmat.vbe, self.cmat.cbe], xmin=0, xmax=self.core_width)
        plt.hlines(
            [self.smat.vbe, self.smat.cbe],
            xmin=self.core_width,
            xmax=self.core_width + self.shell_width,
        )
        lcbe, hcbe = sorted([self.cmat.cbe, self.smat.cbe])
        lvbe, hvbe = sorted([self.cmat.vbe, self.smat.vbe])
        plt.vlines(self.core_width, ymin=lcbe, ymax=hcbe)
        plt.vlines(self.core_width, ymin=lvbe, ymax=hvbe)
        # plt.vlines()
        plt.show()

    # This is current non-normalized.
    def analytical_overlap_integral(self):
        """Computes the analytical electron-hole overlap integral.

        Returns
        -------
        overlap : float
            The electron-hole overlap integral.
        """
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers() * n_
        K_e, Q_e, K_h, Q_h = (
            np.sin(k_e * self.core_width),
            np.sin(q_e * self.shell_width),
            np.sin(k_h * self.core_width),
            np.sin(q_h * self.shell_width),
        )
        norm_e, norm_h = self._normalization()

        R, H = self.core_width, self.shell_width
        core_denom = K_e * K_h * 2 * (k_h * k_h - k_e * k_e)
        shell_denom = Q_e * Q_h * 2 * (q_h * q_h - q_e * q_e)
        # The accompanying formula for these are in a Maxima file.
        # QDWavefunctionsAndIntegrals.wxmx
        core_integral = (
            -(
                (k_h - k_e) * np.sin(R * (k_h + k_e))
                - (k_h + k_e) * np.sin(R * (k_h - k_e))
            )
            * norm_e
            * norm_h
            / core_denom
        )
        shell_integral = (
            -(
                (q_h - q_e) * np.sin(H * (q_h + q_e))
                - (q_h + q_e) * np.sin(H * (q_h - q_e))
            )
            * norm_e
            * norm_h
            / shell_denom
        )

        return abs(core_integral + shell_integral) ** 2

    def numerical_overlap_integral(self):
        """Calculates the numerical electron-hole overlap integral.

        Returns
        -------
        overlap : [float, float]
            The electron-hole overlap integral and a rough estimate of the error in the integral.

        References
        ----------
        .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

        """
        # Scaling to 1 / nm.
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers() * n_
        norm_e, norm_h = self._normalization()

        def ewf(x):
            return wavefunction(x, k_e, q_e, self.core_width, self.shell_width)

        def hwf(x):
            return wavefunction(x, k_h, q_h, self.core_width, self.shell_width)

        def overlap_integrand_real(x):
            return np.real(x * x * ewf(x) * hwf(x))

        def overlap_integrand_imag(x):
            return np.imag(x * x * ewf(x) * hwf(x))

        overlap_integral_real = quad(overlap_integrand_real, 0, self.radius)
        overlap_integral_imag = quad(overlap_integrand_imag, 0, self.radius)
        # Return both the answer and order-of-magnitude of error.
        return (
            abs(
                (overlap_integral_real[0] + 1j * overlap_integral_imag[0])
                * norm_e
                * norm_h
            )
            ** 2,
            (overlap_integral_imag[1] + overlap_integral_real[1]) ** 2,
        )

    def print_e_wf_at_zero(self):
        """Prints the wavefunction at 0."""
        print(_wavefunction(0, self.calculate_wavenumbers()[0], self.core_width))

    # TODO: Implement branch for eh/he coreshells.
    def localization_electron_core(self, shell_width: float = None):
        """Minimum core width for localization of electron for a given shell width.

        Parameters
        ----------
        shell_width : float, nanometers
            The shell width for which to calculate the core localization width. If no value is given, the coreshell's
            current shell width is used.

        Returns
        -------
        localization_width : float, nanometers
            The minimum core localization radius.
        """
        if self.type_one:
            raise NotImplementedError
        elif self.type_two:
            if self.h_e:
                raise LocalizationNotPossibleError(
                    "Electrons will not localize in the core in h/e structures."
                )

            # EVERYTHING IN THIS FUNCTION HAS BEEN SCALED WITH n_ = 1e-9. There are almost certainly better, more adaptive
            # ways to scale. But for now, the nano- is our lord and saviour.
            if shell_width is None:
                # Scaling to order unity.
                shell_width = self.shell_width

            m = self.cmat.m_e / self.smat.m_e

            # This could use a cached value. This does not change.
            # In the Piryatinski 2007 paper, this is used to set a lower bound on the core radius search bracket.
            # However, I've noticed that this lower bracket often fails. Need to look more into why.
            x1 = brentq(
                _x_residual_function,
                -np.pi + 1e-10,
                0,
                args=(self.cmat.m_e, self.smat.m_e),
            )

            # Same for this.
            # SCALED TO ORDER UNITY.
            k1 = (2 * self.cmat.m_e * m_e * self.ue) ** 0.5 / hbar * n_
            # print('k1', k1, 'x1', x1)
            def min_core_loc_from_shell(r: float) -> float:
                return shell_width + m * r / (1 - m + 1 / tanxdivx(k1 * r))

            if type(x1) == float:
                # print('x1:', x1, 'k1:', k1)
                # print('m-ratio:', m)

                # print('FHigh-:', min_core_loc_from_shell(np.pi / k1 - 1e-4))
                lower_bound, upper_bound = x1 / k1, np.pi / k1
                # print('Low:', lower_bound)
                # print('High:', upper_bound)
                # print("FLow:", min_core_loc_from_shell(lower_bound))
                # print("FHigh:", min_core_loc_from_shell(upper_bound))
                # plt.plot(min_core_loc_from_shell(np.linspace(lower_bound, upper_bound, 1000)))

                # This is the fallback for the case of where the sign doesn't change, and we have to drop the lower
                # limit to 0.
                if (
                    min_core_loc_from_shell(lower_bound)
                    * min_core_loc_from_shell(upper_bound)
                    > 0
                ):  # No sign change.
                    # plt.plot(min_core_loc_from_shell(np.linspace(lower_bound, upper_bound, 1000)))
                    # plt.show()

                    # TODO: This lower bound does not agree with the paper. Need to figure this garbage out.
                    lower_bound, upper_bound = scan_and_bracket(
                        min_core_loc_from_shell, 0, upper_bound, 10000
                    )
                    # print('FALLBACKLOW:', lower_bound)
                    # print('FALLBACKHIGH:', upper_bound)
                    # print("FBFLOW:", min_core_loc_from_shell(lower_bound))
                    # print('FLow+:', min_core_loc_from_shell(x1 / k1 + 1e-4))
                    # print("FBFHIGH:", min_core_loc_from_shell(upper_bound))

                result = brentq(min_core_loc_from_shell, lower_bound, upper_bound)

                # Returning with proper scaling.
                return result
            else:
                raise ValueError

    # TODO: Implement branch for eh/he coreshells.
    def localization_hole_shell(self, core_width: float = None):
        """Minimum core width for localization of electron for a given shell width.

        Parameters
        ----------
        core_width : float, nanometers
            The core width for which to calculate the shell localization width. If no value is given, the coreshell's
            current core width is used.

        Returns
        -------
        float, nanometers. : The minimum shell localization radius.

        """
        if self.h_e:
            raise LocalizationNotPossibleError(
                "Holes will not localize in the shell in e/h structures."
            )

        # EVERYTHING IN THIS FUNCTION HAS BEEN SCALED WITH n_ = 1e-9. There are almost certainly better, more adaptive
        # ways to scale. But for now, the nano- is our lord and saviour.
        if core_width is None:
            # Scaling to order unity.
            core_width = self.core_width

        # This could use a cached value. This does not change.
        # In the Piryatinski 2007 paper, this is used to set a lower bound on the core radius search bracket.
        # However, I've noticed that this lower bracket often fails. Need to look more into why.

        # Same for this.
        # SCALED TO ORDER UNITY.
        q1 = (2 * self.smat.m_h * m_e * self.uh) ** 0.5 / hbar * n_
        # print('k1', k1, 'x1', x1)
        def min_shell_loc_from_core(h: float) -> float:
            return core_width + np.tan(q1 * h) * q1

            # print('FALLBACKLOW:', lower_bound)
            # print('FALLBACKHIGH:', upper_bound)
            # print("FBFLOW:", min_core_loc_from_shell(lower_bound))
            # print('FLow+:', min_core_loc_from_shell(x1 / k1 + 1e-4))
            # print("FBFHIGH:", min_core_loc_from_shell(upper_bound))

        result = brentq(
            min_shell_loc_from_core, np.pi / (2 * q1) + 1e-13, np.pi / q1 - 1e-13
        )
        return result


    def coulomb_screening_energy(self, relative_tolerance: float = 1e-4):
        """ Calculates the Coulomb screening energy. Somewhat slow.

        Parameters
        ----------
        relative_tolerance: float
            The relative tolerance for the Coulomb screening energy integral. Defaults to 1e-3.

        Returns
        -------
        2-array of floats: The Coulomb screening energy and error.
        """
        coulomb_screening_operator = make_coulomb_screening_operator(self)
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers() * n_
        # print(k_e, k_h, q_e, q_h)
        norm_e, norm_h = self._normalization()
        # Electron/hole density functions.
        def edf(x):
            return (
                abs(_wavefunction(x, k_e, q_e, self.core_width, self.shell_width)) ** 2
            )

        def hdf(x):
            return (
                abs(_wavefunction(x, k_h, q_h, self.core_width, self.shell_width)) ** 2
            )

        coulomb_integrand = (
            lambda r1, r2: r1 ** 2
            * r2 ** 2
            * edf(r1)
            * hdf(r2)
            * coulomb_screening_operator(r1, r2)
        )

        # Energy returned in units of eV.
        return (
            np.array(
                dblquad(
                    coulomb_integrand,
                    0,
                    self.radius,
                    0,
                    self.radius,
                    epsrel=relative_tolerance,
                )
            )
            * norm_e
            * norm_h
        )

    def interface_polarization_energy(self, relative_tolerance: float = 1e-4):
        """

        Parameters
        ----------
        relative_tolerance

        Returns
        -------

        """
        interface_polarization_operator = make_interface_polarization_operator(self)

        k_e, q_e, k_h, q_h = self.calculate_wavenumbers() * n_
        norm_e, norm_h = self._normalization()
        # print("L484: norms", norm_e, norm_e)
        # Electron/hole density functions.
        def edf(x):
            return (
                abs(_wavefunction(x, k_e, q_e, self.core_width, self.shell_width)) ** 2
            )

        def hdf(x):
            return (
                abs(_wavefunction(x, k_h, q_h, self.core_width, self.shell_width)) ** 2
            )

        def polarization_integrand(r1, r2):
            return (
                r1 ** 2
                * r2 ** 2
                * edf(r1)
                * hdf(r2)
                * interface_polarization_operator(r1, r2)
            )

        # print("L504, wnums", k_e, q_e, k_h, q_h)
        return (
            np.array(
                dblquad(
                    polarization_integrand,
                    0,
                    self.radius,
                    0,
                    self.radius,
                    epsrel=relative_tolerance,
                )
            )
            * norm_e
            * norm_h
        )

    # This is likely to get refactored later to return types.
    def _is_type_one(self):
        return (self.cmat.vbe > self.smat.vbe) and (self.cmat.cbe < self.smat.cbe)

    def _is_type_two(self):
        """"A type two QD has both conduction and valence band edges of its core either higher or lower than the
        corresponding band edges of the shell."""
        core_higher = (self.cmat.vbe > self.smat.vbe) and (
            self.cmat.cbe > self.smat.cbe
        )
        shell_higher = (self.cmat.vbe < self.smat.vbe) and (
            self.cmat.cbe < self.smat.cbe
        )
        return core_higher or shell_higher, core_higher, shell_higher

    def _normalization(self):

        if self.norm_valid:
            return self.norm_e, self.norm_h

        else:
            k_e, q_e, k_h, q_h = self.calculate_wavenumbers() * n_
            # print(k_h)
            electron_density_integral = (
                4
                * np.pi
                * quad(
                    lambda x: x
                    * x
                    * _densityfunction(x, k_e, q_e, self.core_width, self.shell_width),
                    0,
                    self.radius,
                )[0]
            )
            hole_density_integral = (
                4
                * np.pi
                * quad(
                    lambda x: x
                    * x
                    * _densityfunction(x, k_h, q_h, self.core_width, self.shell_width),
                    0,
                    self.radius,
                )[0]
            )

            self.norm_e = 1 / electron_density_integral
            self.norm_h = 1 / hole_density_integral
            self.norm_valid = True
            return self.norm_e, self.norm_h
