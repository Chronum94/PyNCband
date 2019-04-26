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

        self.name = core_material.name + "/" + shell_material.name

        self.cmat = core_material
        self.smat = shell_material

        # I'll see if we need to scale these here. If all our calculations use scaled lengths,
        # we can simply work with nanometers.

        self.core_width = core_width
        self.shell_width = shell_width
        self.radius = core_width + shell_width

        self.type_one, self.type_one_reverse = self._is_type_one()

        # Need to refactor this method/
        self.type_two, self.h_e, self.e_h = self._is_type_two()
        # This is an observer variable so we don't have to recalculate eigen-energies every time
        # unless core/shell dimensions change.
        self.energies_valid: bool = False
        self.s1_e, self.s1_h = None, None

        # Observer variable for normalization constant of wavefunction.
        self.norm_valid: bool = False
        self.norm_e, self.norm_h = None, None

        # All these energies are in eV. All internal calculations are carried out in eV.
        self.ue = np.abs(self.cmat.cbe - self.smat.cbe)
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe)

        self.bandgap = min(self.cmat.cbe, self.smat.cbe) - max(
            self.cmat.vbe, self.smat.vbe
        )

        self.environment_epsilon = environment_epsilon

        # This is used to refine the scanning of energies.
        # For coreshells with massive disparities, the energy scan_and_bracket needs 'adaptive' refinement.
        self.scan_refinement_multiplier = max(core_width, shell_width) / min(
            core_width, shell_width
        )

        self.BASE_ENERGY_SCAN_RESOLUTION = 500
        self.MAX_ENERGY_BRACKETING_ATTEMPTS = 500
        self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV = 0.1
        self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV = 0.1

    def set_core_width(self, x: float):
        """Sets the core radius to the new value. This invalidates calculated energies and wavenumbers, and when needed,
        the values are recalculated.

        Parameters
        ----------
        x : float, nanometers
            The new core width.

        Returns
        -------

        """
        self.radius = self.radius - self.core_width + x
        self.core_width = x
        self.energies_valid = False
        self.norm_valid = False

    def set_shell_width(self, x: float):
        """Sets the shell thickness to the new value. This invalidates calculated energies and wavenumbers, and when
        needed, the values are recalculated.

        Parameters
        ----------
        x : float, nanometers
            The new shell width.

        Returns
        -------

        """
        self.radius = self.radius - self.shell_width + x
        self.shell_width = x
        self.energies_valid = False
        self.norm_valid = False

    def calculate_wavenumbers(self) -> np.ndarray:
        """Returns a 1D 4-element Numpy array of wavenumbers in the core/shell for electrons and holes respectively.

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
        # _e for electrons, _h for holes.
        energy_e, energy_h = self.calculate_s1_energies()

        if self.type_one:
            return np.array(
                [
                    wavenumber_from_energy(energy_e, self.cmat.m_e),
                    wavenumber_from_energy(
                        energy_e, self.smat.m_e, potential_offset=self.ue
                    ),
                    wavenumber_from_energy(energy_h, self.cmat.m_h),
                    wavenumber_from_energy(
                        energy_h, self.smat.m_h, potential_offset=self.uh
                    ),
                ]
            )
        elif self.type_one_reverse:
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

    # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable
    def calculate_s1_energies(self, resolution: int = None) -> np.ndarray:
        """Calculates eigenenergies of the S1 exciton state in eV.

        This may be changed in future versions to calculate higher energy states.

        Parameters
        ----------

        resolution : int
            The number of points to use when scanning and bracketing.

        Returns
        -------
        s1_energies : 2-array of float, eV
            The s1 state energies of electrons and holes.

        """

        # This is a heuristic way to scan finer points for larger quantum dots since their energies are lower.
        if resolution is None:
            resolution = int(
                self.BASE_ENERGY_SCAN_RESOLUTION * self.scan_refinement_multiplier
            )

        # _e for electrons, _h for holes.
        lower_bound_e = 0
        upper_bound_e = self.DEFAULT_ELECTRON_ENERGY_SEARCH_RANGE_EV
        lower_bound_h = 0
        upper_bound_h = self.DEFAULT_HOLE_ENERGY_SEARCH_RANGE_EV

        # Energy brackets.
        electron_bracket_found, hole_bracket_found = False, False
        current_electron_bracketing_attempt, current_hole_bracketing_attempt = 0, 0

        # Electron eigenvalue residual, defined here so we don't have to pass in 'self' in the rootfinder.
        # Might be worth testing for speed/performance.
        # TODO: Get rid of this intermediate currying and alter scan_and_bracket accordingly for all instances.
        def eer(x):
            return electron_eigenvalue_residual(x, self)

        while (
            not electron_bracket_found
            and current_electron_bracketing_attempt
            <= self.MAX_ENERGY_BRACKETING_ATTEMPTS
        ):
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
        # Use the bracket to find the root.
        self.s1_e = brentq(eer, bracket_low, bracket_high)

        # Hole eigenvalue residual.
        def her(x):
            return hole_eigenvalue_residual(x, self)

        while (
            not hole_bracket_found
            and current_hole_bracketing_attempt <= self.MAX_ENERGY_BRACKETING_ATTEMPTS
        ):
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

        # Use bracket, find root.
        self.s1_h = brentq(her, bracket_low, bracket_high)

        self.energies_valid = True

        return np.array([self.s1_e, self.s1_h])

    def plot_electron_wavefunction(self, resolution: int = 1000):
        """

        Parameters
        ----------

        resolution : int
            The resolution of the 1D wavefunction raster.

        Returns
        -------
        x : 1D array of floats
            The radial positions along which the wavefunction is sampled.

        y : 1D array of floats
            The value of the wavefunction, normalized to a peak absolute value of 1.

        """
        core_wavenumber, shell_wavenumber, _, _ = self.calculate_wavenumbers()
        x = np.linspace(0, self.radius, resolution)
        y = wavefunction(
            x, core_wavenumber, shell_wavenumber, self.core_width, self.shell_width
        )
        return x, y / np.max(np.abs(y))

    def plot_hole_wavefunction(self):
        """

        Parameters
        ----------

        resolution : int
            The resolution of the 1D wavefunction raster.

        Returns
        -------
        x : 1D array of floats
            The radial positions along which the wavefunction is sampled.

        y : 1D array of floats
            The value of the wavefunction, normalized to a peak absolute value of 1.

        """
        _, _, core_wavenumber, shell_wavenumber = self.calculate_wavenumbers()
        x = np.linspace(0, self.radius, 1000)
        y = wavefunction(
            x, core_wavenumber, shell_wavenumber, self.core_width, self.shell_width
        )
        return x, y / np.max(np.abs(y))

    def plot_potential_profile(self):
        """Plots one half of the spherically symmetric potential well of the quantum dot.

        Returns
        -------

        """
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
        plt.show()

    # This is current non-normalized.
    def analytical_overlap_integral(self):
        """Computes the analytical electron-hole overlap integral.

        Returns
        -------
        overlap : float
            The electron-hole overlap integral.

        References
        ----------
        .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

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
        core_integral = (
            -(
                (k_h - k_e) * np.sin(R * (k_h + k_e))
                - (k_h + k_e) * np.sin(R * (k_h - k_e))
            )
            / core_denom
        )
        shell_integral = (
            -(
                (q_h - q_e) * np.sin(H * (q_h + q_e))
                - (q_h + q_e) * np.sin(H * (q_h - q_e))
            )
            / shell_denom
        )
        return abs(core_integral + shell_integral) ** 2 * norm_h * norm_e

    def numerical_overlap_integral(self):
        """Calculates the numerical electron-hole overlap integral.

        Returns
        -------
        overlap : [float, float]
            The electron-hole overlap integral. The absolute error is bounded by 1.49e-8. Tolerance bounds may be a
            user option in the future.

        References
        ----------
        .. [1] Piryatinski, A., Ivanov, S. A., Tretiak, S., & Klimov, V. I. (2007). Effect of Quantum and Dielectric \
        Confinement on the Exciton−Exciton Interaction Energy in Type II Core/Shell Semiconductor Nanocrystals. \
        Nano Letters, 7(1), 108–115. https://doi.org/10.1021/nl0622404

        """
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
        return (
            abs((overlap_integral_real[0] + 1j * overlap_integral_imag[0])) ** 2
            * norm_e
            * norm_h
        )

    def localization_electron_core(
        self, shell_width: float = None, asymp: bool = False
    ) -> float:
        """Minimum core width for localization of electron for a given shell width.

        Electrons will only localize in the core in Type 1 core-shells, and Type 2 e/h core-shells.

        Parameters
        ----------
        shell_width : float, nanometers
            The shell width for which to calculate the core localization width. If no value is given, the coreshell's
            current shell width is used.

        asymp: bool
            If true, returns the asymptotic core radius for large shell thickness.

        Returns
        -------
        localization_width : float, nanometers
            The minimum core radius for electron localization.

        """
        if self.type_one:
            raise NotImplementedError
        elif self.type_two:
            if self.h_e:
                raise LocalizationNotPossibleError(
                    "Electrons will not localize in the core in h/e structures."
                )

            if shell_width is None:
                shell_width = self.shell_width

            mass_ratio_coreshell = self.cmat.m_e / self.smat.m_e

            minimum_size_parameter: float = brentq(
                minimum_core_localization_size_parameter,
                0,
                np.pi - 1e-8,
                args=(self.cmat.m_e, self.smat.m_e),
            )

            threshold_core_wavenumber = (
                (2 * self.cmat.m_e * m_e * self.ue) ** 0.5
                / hbar_ev
                * wavenumber_nm_from_energy_ev
            )

            if asymp:
                return minimum_size_parameter / threshold_core_wavenumber

            def min_core_loc_from_shell(r: float) -> float:
                # This is correct. This has been fixed from the paper.
                return shell_width + mass_ratio_coreshell * r / (
                    -1
                    + mass_ratio_coreshell
                    + 1 / tanxdivx(threshold_core_wavenumber * r)
                )

            # The small offsets are so the floating point negatives/positives do not overflow, and the bracketing
            # works correctly. This will work fine unless we have a core/shell that tens of thousands of times thicker/thinner than
            # the shell/core and/or we have a carrier that's tens of thousands of times heavier/lighter in one material than the other.
            lower_bound, upper_bound = (
                minimum_size_parameter / threshold_core_wavenumber + 1e-8,
                np.pi / threshold_core_wavenumber - 1e-8,
            )

            result: float = brentq(min_core_loc_from_shell, lower_bound, upper_bound)

            return result

    def localization_electron_shell(
        self, core_width: float = None, asymp: bool = False
    ) -> float:
        """Minimum shell width for localization of electron for a given core width.

        Parameters
        ----------
        core_width : float, nanometers
            The core width for which to calculate the shell localization width. If no value is given, the coreshell's
            current core width is used.

        asymp : bool
            If true, returns the asymptotic minimum core radius for electron localization, for large shell thicknesses.
        Returns
        -------
        float, nanometers. : The minimum shell localization radius.

        """
        if self.e_h:
            raise LocalizationNotPossibleError(
                "Electrons will not localize in the shell in e/h structures."
            )

        if core_width is None:
            core_width = self.core_width

        theshold_shell_wavenumber = (
            (2 * self.smat.m_e * m_e * self.ue) ** 0.5
            / hbar_ev
            * wavenumber_nm_from_energy_ev
        )

        if asymp:
            return np.pi / (2 * theshold_shell_wavenumber)

        # print('k1', k1, 'x1', x1)
        def min_shell_loc_from_core(h: float) -> float:
            return (
                core_width
                + np.tan(theshold_shell_wavenumber * h) / theshold_shell_wavenumber
            )

        # The small offsets are so the floating point negatives/positives do not overflow, and the bracketing
        # works correctly. This will work fine unless we have a core/shell that tens of thousands of times thicker/thinner than
        # the shell/core and/or we have a carrier that's tens of thousands of times heavier/lighter in one material than the other.
        result: float = brentq(
            min_shell_loc_from_core,
            np.pi / (2 * theshold_shell_wavenumber) + 1e-8,
            np.pi / theshold_shell_wavenumber - 1e-8,
        )
        return result

    def localization_hole_core(
        self, shell_width: float = None, resolution=1000, asymp: bool = False
    ) -> float:
        """Minimum core width for localization of holes for a given shell width.

        Parameters
        ----------
        shell_width : float, nanometers
            The shell width for which to calculate the core localization width. If no value is given, the coreshell's
            current shell width is used.

        asymp : bool
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
                raise LocalizationNotPossibleError(
                    "Holes will not localize in the core in e/h structures."
                )

            if shell_width is None:
                shell_width = self.shell_width

            mass_ratio_coreshell = self.cmat.m_h / self.smat.m_h

            minimum_size_parameter: float = brentq(
                minimum_core_localization_size_parameter,
                0,
                np.pi - 1e-10,
                args=(self.cmat.m_h, self.smat.m_h),
            )

            theshold_core_wavenumber = (
                (2 * self.cmat.m_h * m_e * self.uh) ** 0.5
                / hbar_ev
                * wavenumber_nm_from_energy_ev
            )

            if asymp:
                return minimum_size_parameter / theshold_core_wavenumber

            def min_core_loc_from_shell(r: float) -> float:
                # Fixed from the paper.
                return shell_width + mass_ratio_coreshell * r / (
                    -1
                    + mass_ratio_coreshell
                    + 1 / tanxdivx(theshold_core_wavenumber * r)
                )

            # The small offsets are so the floating point negatives/positives do not overflow, and the bracketing
            # works correctly. This will work fine unless we have a core/shell that tens of thousands of times thicker/thinner than
            # the shell/core and/or we have a carrier that's tens of thousands of times heavier/lighter in one material than the other.
            lower_bound, upper_bound = (
                minimum_size_parameter / theshold_core_wavenumber + 1e-8,
                np.pi / theshold_core_wavenumber - 1e-8,
            )
            result: float = brentq(min_core_loc_from_shell, lower_bound, upper_bound)

            return result

    def localization_hole_shell(
        self, core_width: float = None, asymp: bool = False
    ) -> float:
        """Minimum core width for localization of hole for a given shell width.

        Parameters
        ----------
        core_width : float, nanometers
            The core width for which to calculate the shell localization width. If no value is given, the coreshell's
            current core width is used.

        asymp : bool
            If true, returns the asymptotic minimum shell thickness for hole localization, for large core radii.
        Returns
        -------
        float, nanometers. : The minimum shell localization radius.

        """
        if self.h_e:
            raise LocalizationNotPossibleError(
                "Holes will not localize in the shell in h/e structures."
            )

        if core_width is None:
            core_width = self.core_width

        theshold_shell_wavenumber = wavenumber_from_energy(self.uh, self.smat.m_h)

        if asymp:
            return np.pi / (2 * theshold_shell_wavenumber)

        def min_shell_loc_from_core(h: float) -> float:
            return (
                core_width
                + np.tan(theshold_shell_wavenumber * h) / theshold_shell_wavenumber
            )

        # The small offsets are so the floating point negatives/positives do not overflow, and the bracketing
        # works correctly. This will work fine unless we have a core/shell that tens of thousands of times thicker/thinner than
        # the shell/core and/or we have a carrier that's tens of thousands of times heavier/lighter in one material than the other.
        result: float = brentq(
            min_shell_loc_from_core,
            np.pi / (2 * theshold_shell_wavenumber) + 1e-8,
            np.pi / theshold_shell_wavenumber - 1e-8,
        )
        return result

    def coulomb_screening_energy(
        self,
        relative_tolerance: float = 1e-5,
        shell_term_denominator: float = 2.0
    ):
        """ Calculates the Coulomb screening energy. Somewhat slow.

        Parameters
        ----------
        relative_tolerance : float
            The relative tolerance for the Coulomb screening energy integral. Defaults to 1e-5.

        plot_integrand : bool

        cmap : str

        shell_term_denominator : float
            Equation 8 in Piryatinski/Klimov NanoLett '07 has a factor of two in the second term's denominator.
            However, the model matches up to experiment closer when this term is 1. For correctness, I've kept it
            default to 0, but it is possible that there might be wrong (from the original literature) somewhere.

        Returns
        -------
        2-array of floats: The Coulomb screening energy and error.

        """
        coulomb_screening_operator = make_coulomb_screening_operator(self, shell_term_denominator=shell_term_denominator)
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
        norm_e, norm_h = self._normalization()

        # Electron/hole density functions.
        def edf(x):
            return _densityfunction(x, k_e, q_e, self.core_width, self.shell_width)

        def hdf(x):
            return _densityfunction(x, k_h, q_h, self.core_width, self.shell_width)

        coulomb_integrand = (
            lambda r1, r2: r1 ** 2
            * r2 ** 2
            * edf(r1)
            * hdf(r2)
            * coulomb_screening_operator(r1, r2)
        )

        # Energy returned in units of eV.
        # r1 < R, r2 < R
        region_one = np.array(
            dblquad(
                coulomb_integrand,
                0,
                self.core_width,
                0,
                self.core_width,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        # r1 > R, r2 < R

        region_two = np.array(
            dblquad(
                coulomb_integrand,
                self.core_width,
                self.radius,
                0,
                self.core_width,
                epsrel=relative_tolerance,
            )
        )

        # r1 > R, r2 > R
        region_three = np.array(
            dblquad(
                coulomb_integrand,
                self.core_width,
                self.radius,
                self.core_width,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        # r1 < R, r2 > R
        region_four = np.array(
            dblquad(
                coulomb_integrand,
                0,
                self.core_width,
                self.core_width,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )
        sectioned_integral = (
            (region_one + region_two + region_three + region_four) * norm_h * norm_e
        )

        # DO NOT DELETE THIS CODE. THIS CODE IS A TESTAMENT TO THE LIMITATIONS OF QUADRATURE ALGORITHMS.
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
        #
        #
        # ALSO THIS. THIS IS A ROMBERG INTEGRAL TO SHOW US THAT THE PIECEWISE APPROACH IS CORRECT.
        # trapzed = romb(romb(zz)) * dr * dr * norm_e * norm_h
        # print(whole_integral[0], sectioned_integral[0], trapzed)
        return sectioned_integral

    def polarization_screening_energy(
        self,
        relative_tolerance: float = 1e-5
    ):
        """

        Parameters
        ----------
        relative_tolerance : float
        plot_integrand : bool
        cmap : str

        Returns
        -------

        """

        interface_polarization_operator = make_interface_polarization_operator(self)

        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
        norm_e, norm_h = self._normalization()

        # Electron/hole density functions.
        def edf(x):
            return _densityfunction(x, k_e, q_e, self.core_width, self.shell_width)

        def hdf(x):
            return _densityfunction(x, k_h, q_h, self.core_width, self.shell_width)

        def polarization_integrand(r1, r2):
            return (
                r1 ** 2
                * r2 ** 2
                * edf(r1)
                * hdf(r2)
                * interface_polarization_operator(r1, r2)
            )

        # r1 < R, r2 < R
        integral_region_one = np.array(
            dblquad(
                polarization_integrand,
                0,
                self.core_width,
                0,
                self.core_width,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        # r1 > R, r2 < R
        integral_region_two = np.array(
            dblquad(
                polarization_integrand,
                self.core_width,
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
                self.core_width,
                self.radius,
                self.core_width,
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
                self.core_width,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        sectioned_integral = (
            (
                integral_region_one
                + integral_region_two
                + integral_region_three
                + integral_region_four
            )
            * norm_e
            * norm_h
        )

        # DO NOT DELETE THIS CODE. THIS CODE IS A TESTAMENT TO THE LIMITATIONS OF QUADRATURE ALGORITHMS.
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

        return sectioned_integral

    def self_interaction_energy(
        self,
        relative_tolerance: float = 1e-5,
        plot_integrand: bool = False,
        cmap: str = "coolwarm",
    ):
        """

        Parameters
        ----------
        relative_tolerance : float
        plot_integrand : bool
        cmap : str

        Returns
        -------

        """

        self_interaction_operator = make_self_interaction_operator(self)

        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
        norm_e, norm_h = self._normalization()

        # Electron/hole density functions.
        def edf(x):
            return _densityfunction(x, k_e, q_e, self.core_width, self.shell_width)

        def hdf(x):
            return _densityfunction(x, k_h, q_h, self.core_width, self.shell_width)

        def electron_self_energy_integrand(r):
            return r ** 2 * edf(r) * self_interaction_operator(r)

        def hole_self_energy_integrand(r):
            return r ** 2 * hdf(r) * self_interaction_operator(r)

        integrand_in_core_electron = quad(
            electron_self_energy_integrand,
            0,
            self.core_width,
            epsabs=0.0,
            epsrel=relative_tolerance,
        )
        integrand_in_shell_electron = quad(
            electron_self_energy_integrand,
            self.core_width,
            self.radius,
            epsabs=0.0,
            epsrel=relative_tolerance,
        )

        integrand_in_core_hole = quad(
            hole_self_energy_integrand,
            0,
            self.core_width,
            epsabs=0.0,
            epsrel=relative_tolerance,
        )
        integrand_in_shell_hole = quad(
            hole_self_energy_integrand,
            self.core_width,
            self.radius,
            epsabs=0.0,
            epsrel=relative_tolerance,
        )
        return (
            integrand_in_core_electron[0] + integrand_in_shell_electron[0]
        ) * norm_e + (integrand_in_core_hole[0] + integrand_in_shell_hole[0]) * norm_h

    def biexciton_coulomb_screening_energy(
        self,
        relative_tolerance: float = 1e-5,
        plot_integrand: bool = False,
        cmap: str = "coolwarm",
    ):
        """ Calculates the Coulomb screening energy. Somewhat slow.

        Parameters
        ----------
        relative_tolerance : float
            The relative tolerance for the Coulomb screening energy integral. Defaults to 1e-5.

        plot_integrand : bool

        cmap : str

        Returns
        -------
        2-array of floats: The Coulomb screening energy and error.

        """
        coulomb_screening_operator = make_coulomb_screening_operator(self)
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
        norm_e, norm_h = self._normalization()

        # Carrier difference density function.
        def cddf(x):
            return norm_e * _densityfunction(
                x, k_e, q_e, self.core_width, self.shell_width
            ) - norm_h * _densityfunction(
                x, k_h, q_h, self.core_width, self.shell_width
            )

        coulomb_integrand = (
            lambda r1, r2: r1 ** 2
            * r2 ** 2
            * cddf(r1)
            * coulomb_screening_operator(r1, r2)
            * (-cddf(r2))
        )

        # Energy returned in units of eV.
        # r1 < R, r2 < R
        region_one = np.array(
            dblquad(
                coulomb_integrand,
                0,
                self.core_width,
                0,
                self.core_width,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        # r1 > R, r2 < R

        region_two = np.array(
            dblquad(
                coulomb_integrand,
                self.core_width,
                self.radius,
                0,
                self.core_width,
                epsrel=relative_tolerance,
            )
        )

        # r1 > R, r2 > R
        region_three = np.array(
            dblquad(
                coulomb_integrand,
                self.core_width,
                self.radius,
                self.core_width,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )

        # r1 < R, r2 > R
        region_four = np.array(
            dblquad(
                coulomb_integrand,
                0,
                self.core_width,
                self.core_width,
                self.radius,
                epsabs=0.0,
                epsrel=relative_tolerance,
            )
        )
        sectioned_integral = region_one + region_two + region_three + region_four

        if plot_integrand:
            r, dr = np.linspace(1e-13, self.radius, 128, retstep=True)
            r1, r2 = np.meshgrid(r, r)
            coulomb_integrand = np.vectorize(coulomb_integrand)
            max_core_sample = r[np.argwhere(r < self.core_width)[-1]]
            zz = coulomb_integrand(r1, r2)
            plt.imshow(zz, extent=[0, self.radius, self.radius, 0], cmap=cmap)
            plt.hlines(
                max_core_sample,
                xmin=0,
                xmax=self.radius,
                linestyles="dotted",
                label="H-shell",
                linewidth=0.5,
            )
            plt.vlines(
                max_core_sample,
                ymin=0,
                ymax=self.radius,
                linestyles="dotted",
                label="V-core",
                linewidth=0.5,
            )
            plt.colorbar()
            plt.xlabel("Electron($r_a$) coordinate")
            plt.ylabel("Hole($r_b$) coordinate")
            plt.title("Coulomb integrand")
            plt.show()
        #
        #
        # ALSO THIS. THIS IS A ROMBERG INTEGRAL TO SHOW US THAT THE PIECEWISE APPROACH IS CORRECT.
        # trapzed = romb(romb(zz)) * dr * dr * norm_e * norm_h
        # print(whole_integral[0], sectioned_integral[0], trapzed)
        return sectioned_integral

    # This is likely to get refactored later to return types.
    def _is_type_one(self) -> (bool, bool):
        is_type_one = (self.cmat.vbe > self.smat.vbe) and (
            self.cmat.cbe < self.smat.cbe
        )
        is_reverse_type_one = (self.cmat.vbe < self.smat.vbe) and (
            self.cmat.cbe > self.smat.cbe
        )
        return is_type_one, is_reverse_type_one

    def _is_type_two(self) -> (bool, bool, bool):
        """"A type two QD has both conduction and valence band edges of its core either higher or lower than the
        corresponding band edges of the shell.

        """
        core_higher = (self.cmat.vbe > self.smat.vbe) and (
            self.cmat.cbe > self.smat.cbe
        )
        shell_higher = (self.cmat.vbe < self.smat.vbe) and (
            self.cmat.cbe < self.smat.cbe
        )
        return core_higher or shell_higher, core_higher, shell_higher

    def _normalization(self) -> (float, float):

        if self.norm_valid:
            return self.norm_e, self.norm_h

        else:
            k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
            # print(k_h)
            electron_density_integral = quad(
                lambda x: x
                * x
                * _densityfunction(x, k_e, q_e, self.core_width, self.shell_width),
                0,
                self.radius,
            )[0]
            hole_density_integral = quad(
                lambda x: x
                * x
                * _densityfunction(x, k_h, q_h, self.core_width, self.shell_width),
                0,
                self.radius,
            )[0]

            self.norm_e = 1 / electron_density_integral
            self.norm_h = 1 / hole_density_integral
            self.norm_valid = True
            return self.norm_e, self.norm_h

    def __str__(self):
        return self.name
