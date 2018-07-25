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

__all__ = ["CoreShellParticle"]


class CoreShellParticle:
    def __init__(
        self,
        core_material: Material,
        shell_material: Material,
        core_width: float,
        shell_width: float,
        environment_epsilon: float = 1.0,
    ):
        """Creates a core-shell nanoparticle.

        Parameters
        ----------
        core_material : Material
            A Material object representing the core material of the nanocrystal.
        shell_material : Material
            A Material object representing the shell material of the nanocrystal.
        core_width : float, nanometers
        shell_width : float, nanometers

        """

        # Preliminary asserts.
        assert isinstance(core_material, Material)
        assert isinstance(shell_material, Material)
        assert core_width is not None
        assert shell_width is not None

        self.cmat = core_material
        self.smat = shell_material

        # I'll see if we need to scale these here. If all our calculations use scaled lengths,
        # we can simply work with nanometers.

        self.core_width = core_width
        self.shell_width = shell_width
        self.radius = core_width + shell_width

        # TODO: Need to add support for reverse type one.
        self.type_one = self._is_type_one()

        # Need to refactor this method/
        self.type_two, self.h_e, self.e_h = self._is_type_two()
        # This is an observer variable so we don't have to recalculate eigen-energies every time.
        self.energies_valid: bool = False
        self.s1_e, self.s1_h = None, None

        # Observer variable for normalization constant of wavefunction.
        self.norm_valid: bool = False
        self.norm_e, self.norm_h = None, None

        # All these energies are in eV. All internal calculations are carried out in eV.
        self.ue = np.abs(self.cmat.cbe - self.smat.cbe)
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe)

        self.bandgap = min(self.cmat.cbe, self.smat.cbe) - max(self.cmat.vbe, self.smat.vbe)

        self.environment_epsilon = environment_epsilon

        # This is used to refine the scanning of energies.
        # For coreshells with massive disparities, the energy scan_and_bracket needs 'adaptive' refinement.
        self.scan_refinement_multiplier = max(core_width, shell_width) / min(core_width, shell_width)

        self.BASE_SCAN_RESOLUTION = 500
        self.MAX_ENERGY_BRACKETING_ATTEMPTS = 500
        self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV = 0.1
        self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV = 0.1

    def set_core_width(self, x: float):
        """

        Parameters
        ----------
        x : float, nanometers
            The new core width.

        Returns
        -------

        """
        self.radius -= self.core_width
        self.core_width = x
        self.radius += self.core_width
        self.energies_valid = False
        self.norm_valid = False

    def set_shell_width(self, x: float):
        """

        Parameters
        ----------
        x : float, nanometers
            The new shell width.

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
        wavenumbers : [float, float, float, float] : Wavenumbers in 1 / nm.
            Array of wavenumbers [electron-core, electron-shell, hole-core, hole-shell]

        References
        ----------
        .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

        .. [2] Li, L., Reiss, P., & Protie, M. (2009). Core / Shell Semiconductor Nanocrystals, (2), 154–168. \
        https://doi.org/10.1002/smll.200800841

        """

        energy_e, energy_h = self.calculate_s1_energies()
        # print('E:', energy_e, energy_h)

        if self.type_one:
            return np.array(
                [
                    wavenumber_from_energy(energy_e, self.cmat.m_e, potential_offset=self.ue),
                    wavenumber_from_energy(energy_e, self.smat.m_e),
                    wavenumber_from_energy(energy_h, self.cmat.m_h, potential_offset=self.uh),
                    wavenumber_from_energy(energy_h, self.smat.m_h),
                ]
            )
        elif self.type_two:
            if self.e_h:
                return np.array(
                    [
                        wavenumber_from_energy(energy_e, self.cmat.m_e),
                        wavenumber_from_energy(energy_e, self.smat.m_e, potential_offset=self.ue),
                        wavenumber_from_energy(energy_h, self.cmat.m_h, potential_offset=self.uh),
                        wavenumber_from_energy(energy_h, self.smat.m_h),
                    ]
                )
            elif self.h_e:
                return np.array(
                    [
                        wavenumber_from_energy(energy_e, self.cmat.m_e, potential_offset=self.ue),
                        wavenumber_from_energy(energy_e, self.smat.m_e),
                        wavenumber_from_energy(energy_h, self.cmat.m_h),
                        wavenumber_from_energy(energy_h, self.smat.m_h, potential_offset=self.uh),
                    ]
                )

    # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable
    def calculate_s1_energies(self, resolution: int = None) -> np.ndarray:
        """Calculates eigenenergies of the S1 exciton state in eV.

        Parameters
        ----------

        resolution : int
            The number of points to use when scanning and bracketing.

        Returns
        -------
        s1_energies : 2-array of float, eV
            The s1 exciton energies of electrons and holes.

        """

        if resolution is None:
            resolution = int(self.BASE_SCAN_RESOLUTION * self.scan_refinement_multiplier)
            # print(resolution)

        lower_bound_e = 0
        upper_bound_e = self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV
        lower_bound_h = 0
        upper_bound_h = self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV

        # Energy brackets.
        electron_bracket_found, hole_bracket_found = (False, False)
        current_electron_bracketing_attempt, current_hole_bracketing_attempt = (0, 0)

        # Electron eigenvalue residual.
        def eer(x):
            return electron_eigenvalue_residual(x, self)

        while not electron_bracket_found and current_electron_bracketing_attempt <= self.MAX_ENERGY_BRACKETING_ATTEMPTS:
            (bracket_low, bracket_high), electron_bracket_found = scan_and_bracket(
                eer, lower_bound_e, upper_bound_e, resolution
            )
            lower_bound_e += self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV
            upper_bound_e += self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV
            current_electron_bracketing_attempt += 1

        if not electron_bracket_found:
            raise EnergyNotBracketedError(
                f"Energy was not bracketed after {self.MAX_ENERGY_BRACKETING_ATTEMPTS} scans "
                f"increasing by {self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV} eV each. Consider "
                "increaseing MAX_ENERGY_BRACKETING_ATTEMPTS or "
                "DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV, or both."
            )
        # print(current_electron_bracketing_attempt)

        self.s1_e = brentq(electron_eigenvalue_residual, bracket_low, bracket_high, args=(self,))

        # Hole eigenvalue residual.
        def her(x):
            return hole_eigenvalue_residual(x, self)

        while not hole_bracket_found and current_hole_bracketing_attempt <= self.MAX_ENERGY_BRACKETING_ATTEMPTS:
            (bracket_low, bracket_high), hole_bracket_found = scan_and_bracket(
                her, lower_bound_h, upper_bound_h, resolution
            )
            lower_bound_h += self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV
            upper_bound_h += self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV
            current_hole_bracketing_attempt += 1

        if not hole_bracket_found:
            raise EnergyNotBracketedError(
                f"Energy was not bracketed after {self.MAX_ENERGY_BRACKETING_ATTEMPTS} scans "
                f"increasing by {self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV} eV each. Consider "
                "increaseing MAX_ENERGY_BRACKETING_ATTEMPTS or "
                "DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV, or both."
            )
        # print(current_hole_bracketing_attempt)

        self.s1_h = brentq(hole_eigenvalue_residual, bracket_low, bracket_high, args=(self,))

        self.energies_valid = True

        return np.array([self.s1_e, self.s1_h])

    def plot_electron_wavefunction(self):
        """

        Returns
        -------

        """
        core_wavenumber, shell_wavenumber, _, _ = self.calculate_wavenumbers()
        x = np.linspace(0, self.radius, 1000)
        y = wavefunction(x, core_wavenumber, shell_wavenumber, self.core_width, self.shell_width)
        return x, y / np.max(np.abs(y))

    def plot_hole_wavefunction(self):
        """

        Returns
        -------

        """
        _, _, core_wavenumber, shell_wavenumber = self.calculate_wavenumbers()
        x = np.linspace(0, self.radius, 1000)
        y = wavefunction(x, core_wavenumber, shell_wavenumber, self.core_width, self.shell_width)
        return x, y / np.max(np.abs(y))

    def plot_potential_profile(self):
        """Plots one half of the spherically symmetric potential well of the quantum dot.

        Returns
        -------

        """
        plt.hlines([self.cmat.vbe, self.cmat.cbe], xmin=0, xmax=self.core_width)
        plt.hlines([self.smat.vbe, self.smat.cbe], xmin=self.core_width, xmax=self.core_width + self.shell_width)
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
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
        K_e, Q_e, K_h, Q_h = (
            np.sin(k_e * self.core_width),
            np.sin(q_e * self.shell_width),
            np.sin(k_h * self.core_width),
            np.sin(q_h * self.shell_width),
        )
        norm_e, norm_h = self._normalization()

        R, H = self.core_width, self.shell_width
        core_denom = K_e * K_h * 2 * (k_h ** 2 - k_e ** 2)
        shell_denom = Q_e * Q_h * 2 * (q_h ** 2 - q_e ** 2)
        # The accompanying formula for these are in a Maxima file.
        # QDWavefunctionsAndIntegrals.wxmx
        core_integral = -((k_h - k_e) * np.sin(R * (k_h + k_e)) - (k_h + k_e) * np.sin(R * (k_h - k_e))) / core_denom
        shell_integral = -((q_h - q_e) * np.sin(H * (q_h + q_e)) - (q_h + q_e) * np.sin(H * (q_h - q_e))) / shell_denom
        # if abs(core_denom) < 1e-2 or abs(shell_denom) < 1e-2:
        #     print(core_denom, shell_denom)
        #     print(R, H)
        #     raise RuntimeWarning("TINY DENOM.")
        return abs(core_integral + shell_integral) ** 2 * norm_h * norm_e

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
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
        norm_e, norm_h = self._normalization()

        def ewf(x):
            return _wavefunction(x, k_e, q_e, self.core_width, self.shell_width)

        def hwf(x):
            return _wavefunction(x, k_h, q_h, self.core_width, self.shell_width)

        def overlap_integrand_real(x):
            return np.real(x * x * ewf(x) * hwf(x))

        def overlap_integrand_imag(x):
            return np.imag(x * x * ewf(x) * hwf(x))

        overlap_integral_real = quad(overlap_integrand_real, 0, self.radius)
        overlap_integral_imag = quad(overlap_integrand_imag, 0, self.radius)
        # Return both the answer and order-of-magnitude of error.
        # print(norm_e, norm_h)
        return (
            abs((overlap_integral_real[0] + 1j * overlap_integral_imag[0])) ** 2
            * norm_e
            * norm_h  # ,
            # (overlap_integral_imag[1] + overlap_integral_real[1]) ** 2,
        )

    def print_e_wf_at_zero(self):
        """Prints the wavefunction at 0."""
        print(_wavefunction(0, self.calculate_wavenumbers()[0], self.core_width))

    def localization_electron_core(self, shell_width: float = None, asymp: bool = False) -> float:
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
                raise LocalizationNotPossibleError("Electrons will not localize in the core in h/e structures.")

            if shell_width is None:
                shell_width = self.shell_width

            m = self.cmat.m_e / self.smat.m_e

            # This could use a cached value. This does not change.
            # In the Piryatinski 2007 paper, this is used to set a lower bound on the core radius search bracket.
            # However, I've noticed that this lower bracket often fails. Need to look more into why.
            x1 = brentq(_x_residual_function, 0, np.pi - 1e-8, args=(self.cmat.m_e, self.smat.m_e))

            k1 = (2 * self.cmat.m_e * m_e * self.ue) ** 0.5 / hbar_ev * wavenumber_nm_from_energy_ev

            if asymp:
                return x1 / k1
            # print('k1', k1, 'x1', x1)
            def min_core_loc_from_shell(r: float) -> float:
                return shell_width + m * r / (1 - m + 1 / tanxdivx(k1 * r))

            if type(x1) == float:
                lower_bound, upper_bound = (x1 / k1 + 1e-8, np.pi / k1 - 1e-8)

                if min_core_loc_from_shell(lower_bound) * min_core_loc_from_shell(upper_bound) > 0:  # No sign change.

                    # warn(
                    #     "Lowering localization search limit. This goes against the paper."
                    # )
                    # TODO: This lower bound does not agree with the paper. Need to figure this garbage out.
                    (lower_bound, upper_bound), bracket_found = scan_and_bracket(
                        min_core_loc_from_shell, 0, upper_bound, 10000
                    )
                result = brentq(min_core_loc_from_shell, lower_bound, upper_bound)

                # Returning with proper scaling.
                return result
            else:
                raise ValueError

    def localization_electron_shell(self, core_width: float = None, asymp: bool = False) -> float:
        """Minimum shell width for localization of electron for a given core width.

        Parameters
        ----------
        core_width : float, nanometers
            The core width for which to calculate the shell localization width. If no value is given, the coreshell's
            current core width is used.

        Returns
        -------
        float, nanometers. : The minimum shell localization radius.

        """
        if self.e_h:
            raise LocalizationNotPossibleError("Electrons will not localize in the shell in e/h structures.")


        if core_width is None:

            core_width = self.core_width


        q1 = (2 * self.smat.m_e * m_e * self.ue) ** 0.5 / hbar_ev * wavenumber_nm_from_energy_ev

        if asymp:
            return np.pi / (2 * q1)
        # print('k1', k1, 'x1', x1)
        def min_shell_loc_from_core(h: float) -> float:
            return core_width + np.tan(q1 * h) / q1

        result = brentq(min_shell_loc_from_core, np.pi / (2 * q1) + 1e-8, np.pi / q1 - 1e-8)
        return result

    def localization_hole_core(self, shell_width: float = None, resolution=1000, asymp: bool = False) -> float:
        """Minimum core width for localization of holes for a given shell width.

        Parameters
        ----------
        shell_width : float, nanometers
            The shell width for which to calculate the core localization width. If no value is given, the coreshell's
            current shell width is used.

        resolution : int
            The resolution with which to look for the roots of the localization equation.

        Returns
        -------
        localization_width : float, nanometers
            The minimum core localization radius.

        """
        if self.type_one:
            raise NotImplementedError
        elif self.type_two:
            if self.e_h:
                raise LocalizationNotPossibleError("Holes will not localize in the core in e/h structures.")

            if shell_width is None:
                shell_width = self.shell_width

            mass_ratio_coreshell = self.cmat.m_h / self.smat.m_h

            # This could use a cached value. This does not change.
            # In the Piryatinski 2007 paper, this is used to set a lower bound on the core radius search bracket.
            # However, I've noticed that this lower bracket often fails. Need to look more into why.
            x1 = brentq(_x_residual_function, 0, np.pi - 1e-10, args=(self.cmat.m_h, self.smat.m_h))


            k1 = (2 * self.cmat.m_h * m_e * self.uh) ** 0.5 / hbar_ev * wavenumber_nm_from_energy_ev

            if asymp:
                return x1 / k1
            # print('k1', k1, 'x1', x1)
            def min_core_loc_from_shell(r: float) -> float:
                return shell_width + mass_ratio_coreshell * r / (1 - mass_ratio_coreshell + 1 / tanxdivx(k1 * r))

            if type(x1) == float:
                # print('x1:', x1, 'k1:', k1)
                # print('m-ratio:', m)

                # print('FHigh-:', min_core_loc_from_shell(np.pi / k1 - 1e-4))
                lower_bound, upper_bound = (x1 / k1 + 1e-8, np.pi / k1 - 1e-8)
                # print('Low:', lower_bound)
                # print('High:', upper_bound)
                # print("FLow:", min_core_loc_from_shell(lower_bound))
                # print("FHigh:", min_core_loc_from_shell(upper_bound))
                # plt.plot(min_core_loc_from_shell(np.linspace(lower_bound, upper_bound, 1000)))

                # This is the fallback for the case of where the sign doesn't change, and we have to drop the lower
                # limit to 0.
                if min_core_loc_from_shell(lower_bound) * min_core_loc_from_shell(upper_bound) > 0:  # No sign change.
                    warn("Pls.")
                    print(self.cmat.m_e)
                    plt.plot(np.linspace(lower_bound, upper_bound, 1000), min_core_loc_from_shell(np.linspace(lower_bound, upper_bound, 1000)))

                    # warn(
                    #     "Lowering localization search limit. This goes against the paper."
                    # )
                    # TODO: This lower bound does not agree with the paper. Need to figure this garbage out.
                    (lower_bound, upper_bound), bracket_found = scan_and_bracket(
                        min_core_loc_from_shell, 0, upper_bound, resolution
                    )
                    plt.plot(np.linspace(lower_bound, upper_bound, 1000),
                             min_core_loc_from_shell(np.linspace(lower_bound, upper_bound, 1000)))
                    plt.show()
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

    def localization_hole_shell(self, core_width: float = None, asymp: bool=False) -> float:
        """Minimum core width for localization of hole for a given shell width.

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
            raise LocalizationNotPossibleError("Holes will not localize in the shell in h/e structures.")

        if core_width is None:
            core_width = self.core_width

        # This could use a cached value. This does not change.
        # In the Piryatinski 2007 paper, this is used to set a lower bound on the core radius search bracket.
        # However, I've noticed that this lower bracket often fails. Need to look more into why.

        q1 = wavenumber_from_energy(self.uh, self.smat.m_h)
        # print("q1 is:", q1)
        if asymp:
            return np.pi / (2 * q1)

        def min_shell_loc_from_core(h: float) -> float:
            return core_width + np.tan(q1 * h) / q1

        result = brentq(min_shell_loc_from_core, np.pi / (2 * q1) + 1e-8, np.pi / q1 - 1e-8)
        return result

    def coulomb_screening_energy(self, relative_tolerance: float = 1e-5, plot_integrand: bool = False, cmap: str='coolwarm'):
        """ Calculates the Coulomb screening energy. Somewhat slow.

        Parameters
        ----------
        relative_tolerance : float
            The relative tolerance for the Coulomb screening energy integral. Defaults to 1e-5.

        plot_integrand : bool

        Returns
        -------
        2-array of floats: The Coulomb screening energy and error.

        """
        coulomb_screening_operator = make_coulomb_screening_operator(self)
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
        # print(k_e, q_e)
        # print(k_e, k_h, q_e, q_h)
        norm_e, norm_h = self._normalization()

        # Electron/hole density functions.
        def edf(x):
            return abs(_wavefunction(x, k_e, q_e, self.core_width, self.shell_width)) ** 2

        def hdf(x):
            return abs(_wavefunction(x, k_h, q_h, self.core_width, self.shell_width)) ** 2

        coulomb_integrand = lambda r1, r2: r1 ** 2 * r2 ** 2 * edf(r1) * hdf(r2) * coulomb_screening_operator(r1, r2)

        piecewise_discontinuity_exclusion = 0.0

        # Energy returned in units of eV.
        # r1 < R, r2 < R
        integral_region_one = np.array(
            dblquad(coulomb_integrand, 0, self.core_width, 0, self.core_width, epsabs=0.0, epsrel=relative_tolerance)
        )

        # r1 > R, r2 < R

        integral_region_two = np.array(
            dblquad(
                coulomb_integrand,
                self.core_width + piecewise_discontinuity_exclusion,
                self.radius,
                0,
                self.core_width,
                epsrel=relative_tolerance,
            )
        )

        # r1 > R, r2 > R
        integral_region_three = np.array(
            dblquad(
                coulomb_integrand,
                self.core_width + piecewise_discontinuity_exclusion,
                self.radius,
                self.core_width + piecewise_discontinuity_exclusion,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        # r1 < R, r2 > R
        integral_region_four = np.array(
            dblquad(
                coulomb_integrand,
                0,
                self.core_width,
                self.core_width + piecewise_discontinuity_exclusion,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )
        sectioned_integral = (
            (integral_region_one + integral_region_two + integral_region_three + integral_region_four) * norm_h * norm_e
        )

        # !!! DO NOT DELETE THIS CODE. THIS CODE IS A TESTAMENT TO THE LIMITATIONS OF QUADRATURE ALGORITHMS.
        # whole_integral = (
        #     np.array(
        #         dblquad(
        #             coulomb_integrand,
        #             0,
        #             self.radius,
        #             0,
        #             self.radius,
        #             epsrel=relative_tolerance,
        #         )
        #     )
        #     * norm_e
        #     * norm_h
        # )
        if plot_integrand:
            r, dr = np.linspace(1e-13, self.radius, 128, retstep=True)
            r1, r2 = np.meshgrid(r, r)
            coulomb_integrand = np.vectorize(coulomb_integrand)
            max_core_sample = r[np.argwhere(r < self.core_width)[-1]]
            zz = coulomb_integrand(r1, r2)
            plt.imshow(zz, extent=[0, self.radius, self.radius, 0], cmap=cmap)
            plt.hlines(max_core_sample, xmin=0, xmax=self.radius, linestyles="dotted", label="H-shell", linewidth=0.5)
            plt.vlines(max_core_sample, ymin=0, ymax=self.radius, linestyles="dotted", label="V-core", linewidth=0.5)
            plt.colorbar()
            plt.xlabel("Electron($r_a$) coordinate")
            plt.ylabel("Hole($r_b$) coordinate")
            plt.title("Coulomb integrand")
            plt.show()
        #
        #
        # !!! ALSO THIS. THIS IS A ROMBERG INTEGRAL TO SHOW US THAT THE PIECEWISE APPROACH IS CORRECT.
        # trapzed = romb(romb(zz)) * dr * dr * norm_e * norm_h
        # print(whole_integral[0], sectioned_integral[0], trapzed)
        return sectioned_integral

    def interface_polarization_energy(self, relative_tolerance: float = 1e-5, plot_integrand: bool = False, cmap: str='coolwarm'):
        """

        Parameters
        ----------
        relative_tolerance
        plot_integrand

        Returns
        -------

        """

        interface_polarization_operator = make_interface_polarization_operator(self)

        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
        norm_e, norm_h = self._normalization()

        # print("L484: norms", norm_e, norm_e)
        # Electron/hole density functions.
        def edf(x):
            return abs(_wavefunction(x, k_e, q_e, self.core_width, self.shell_width)) ** 2

        def hdf(x):
            return abs(_wavefunction(x, k_h, q_h, self.core_width, self.shell_width)) ** 2

        def polarization_integrand(r1, r2):
            return r1 ** 2 * r2 ** 2 * edf(r1) * hdf(r2) * interface_polarization_operator(r1, r2)

        # print("L504, wnums", k_e, q_e, k_h, q_h)
        piecewise_discontinuity_exclusion = 0.0
        # Energy returned in units of eV.
        # r1 < R, r2 < R
        integral_region_one = np.array(
            dblquad(
                polarization_integrand, 0, self.core_width, 0, self.core_width, epsabs=0.0, epsrel=relative_tolerance
            )
        )

        # r1 > R, r2 < R
        integral_region_two = np.array(
            dblquad(
                polarization_integrand,
                self.core_width + piecewise_discontinuity_exclusion,
                self.radius,
                0,
                self.core_width,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        # r1 > R, r2 > R
        integral_region_three = np.array(
            dblquad(
                polarization_integrand,
                self.core_width + piecewise_discontinuity_exclusion,
                self.radius,
                self.core_width + piecewise_discontinuity_exclusion,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        # r1 < R, r2 > R
        integral_region_four = np.array(
            dblquad(
                polarization_integrand,
                0,
                self.core_width,
                self.core_width + piecewise_discontinuity_exclusion,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        sectioned_integral = (
            (integral_region_one + integral_region_two + integral_region_three + integral_region_four) * norm_e * norm_h
        )

        # !!! DO NOT DELETE THIS CODE. THIS CODE IS A TESTAMENT TO THE LIMITATIONS OF QUADRATURE ALGORITHMS.
        # whole_integral = (
        #     np.array(
        #         dblquad(
        #             polarization_integrand,
        #             0,
        #             self.radius,
        #             0,
        #             self.radius,
        #             epsrel=relative_tolerance,
        #         )
        #     )
        #     * norm_e
        #     * norm_h
        # )
        if plot_integrand:
            r, dr = np.linspace(1e-13, self.radius, 256, retstep=True)
            max_core_sample = r[np.argwhere(r < self.core_width)[-1]]
            r1, r2 = np.meshgrid(r, r)
            polarization_integrand = np.vectorize(polarization_integrand)
            zz = polarization_integrand(r1, r2)
            plt.imshow(zz, extent=[0, self.radius, self.radius, 0], cmap=cmap)
            plt.hlines(max_core_sample, xmin=0, xmax=self.radius, linestyles="dotted", label="H-shell", linewidth=0.5)
            plt.vlines(max_core_sample, ymin=0, ymax=self.radius, linestyles="dotted", label="V-core", linewidth=0.5)

            plt.colorbar()
            plt.xlabel("Electron($r_a$) coordinate")
            plt.ylabel("Hole($r_b$) coordinate")
            plt.title("Polarization integrand")
            plt.show()

        # trapzed = romb(romb(zz)) * dr * dr * norm_e * norm_h
        return sectioned_integral

    # This is likely to get refactored later to return types.
    def _is_type_one(self):
        return (self.cmat.vbe > self.smat.vbe) and (self.cmat.cbe < self.smat.cbe)

    def _is_type_two(self):
        """"A type two QD has both conduction and valence band edges of its core either higher or lower than the
        corresponding band edges of the shell."""
        core_higher = (self.cmat.vbe > self.smat.vbe) and (self.cmat.cbe > self.smat.cbe)
        shell_higher = (self.cmat.vbe < self.smat.vbe) and (self.cmat.cbe < self.smat.cbe)
        return core_higher or shell_higher, core_higher, shell_higher

    def _normalization(self):

        if self.norm_valid:
            return self.norm_e, self.norm_h

        else:
            k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
            # print(k_h)
            electron_density_integral = quad(
                lambda x: x * x * _densityfunction(x, k_e, q_e, self.core_width, self.shell_width), 0, self.radius
            )[0]
            hole_density_integral = quad(
                lambda x: x * x * _densityfunction(x, k_h, q_h, self.core_width, self.shell_width), 0, self.radius
            )[0]

            self.norm_e = 1 / electron_density_integral
            self.norm_h = 1 / hole_density_integral
            self.norm_valid = True
            return self.norm_e, self.norm_h
