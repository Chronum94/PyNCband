from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad, dblquad

from scipy.constants import hbar, e, m_e

n_ = 1e-9
from .Material import Material
from .physicsfunctions import *

from numba import jit
# from functools import partial

__all__ = ["CoreShellParticle"]


class CoreShellParticle:
    def __init__(
        self,
        core_material: Material,
        shell_material: Material,
        core_width: float,
        shell_width: float,
    ):

        self.cmat = core_material
        self.smat = shell_material
        self.core_width = core_width * n_
        self.shell_width = shell_width * n_
        self.radius = (core_width + shell_width) *  n_
        self.type_one = self._is_type_one()
        self.type_two, self.h_e, self.e_h = self._is_type_two()
        # This is an observer variable so we don't have to recalculate eigen-energies every time.
        self.energies_valid: bool = False
        self.s1_e, self.s1_h = None, None
        self.ue = np.abs(self.cmat.cbe - self.smat.cbe) * e # Converting to eV
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe) * e

    def set_core_width(self, x):
        self.core_width = x * n_
        self.energies_valid = False

    def set_shell_width(self, x):
        self.shell_width = x * n_
        self.energies_valid = False


    def calculate_wavenumbers(
        self, is_nm = True
    ) -> np.ndarray:
        """Returns a tuple of the electron wavevectors in the core and the shell."""
        # energy_e, energy_h = None, None

        energy_e, energy_h = self.calculate_s1_energies()
        # This gets set to false when we change core/shell radius, etc.

        if self.e_h:
            return np.array([
                wavenumber_from_energy(energy_e, self.cmat.m_e, ),
                wavenumber_from_energy(
                    energy_e, self.smat.m_e, potential_offset=self.ue
                ),
                wavenumber_from_energy(
                    energy_h, self.cmat.m_h, potential_offset=self.uh
                ),
                wavenumber_from_energy(energy_h, self.smat.m_h)])
        elif self.h_e:
            return np.array([
                wavenumber_from_energy(
                    energy_e, self.cmat.m_e, potential_offset=self.ue
                ),
                wavenumber_from_energy(energy_e, self.smat.m_e),
                wavenumber_from_energy(energy_h, self.cmat.m_h),
                wavenumber_from_energy(
                    energy_h, self.smat.m_h, potential_offset=self.uh)])

    # This method can currently only find cases where the energy of the lowest state is above the poetntial step.
    def calculate_s1_energies(self, bounds=(), resolution=1000, as_ev=True) -> Tuple[float, float]:

        if self.energies_valid:
            return self.s1_e, self.s1_h
        else:

            # Bounds in Joules.
            lower_bound_e = 0
            upper_bound_e = 5 * self.ue
            lower_bound_h = 0
            upper_bound_h = 5 * self.uh

            # These bounds are already in Joules. Do not mul by e again.
            x: np.ndarray = np.linspace(lower_bound_e, upper_bound_e, resolution)
            if bounds != ():
                x = np.linspace(bounds[0], bounds[1], resolution)
            ye = electron_eigenvalue_residual(x, self)
            # plt.plot(x, ye)
            # plt.show()
            ye_signs = np.sign(ye)
            ye_sign_change = np.diff(ye_signs)  # This array is one element shorter.
            ye_neg2pos_change = np.argwhere(np.where(ye_sign_change > 0.5, 1, 0))
            root_position = ye_neg2pos_change[0]
            # print(*x[root_position:root_position + 2])
            self.s1_e = brentq(
                electron_eigenvalue_residual,
                x[root_position],
                x[root_position + 1],
                args=(self,),
            )

            x = np.linspace(lower_bound_h, upper_bound_h, resolution)
            if bounds != ():
                x = np.linspace(bounds[2], bounds[3], resolution)
            yh = hole_eigenvalue_residual(x, self)
            yh_signs = np.sign(yh)
            yh_sign_change = np.diff(yh_signs)  # This array is one element shorter.
            yh_neg2pos_change = np.argwhere(np.where(yh_sign_change > 0.5, 1, 0))
            root_position = yh_neg2pos_change[0]
            # print(yh[root_position], yh[root_position + 1])

            self.s1_h = brentq(
                hole_eigenvalue_residual,
                x[root_position],
                x[root_position + 1],
                args=(self,),
            )
            self.energies_valid = True
            if as_ev:
                return self.s1_e / e, self.s1_h / e
            else:
                return self.s1_e, self.s1_h

    def plot_electron_wavefunction(
        self, x, core_wavevector: float, shell_wavevector: float
    ):

        y = wavefunction(
            x, core_wavevector, shell_wavevector, self.core_width, self.shell_width
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
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()
        K_e, Q_e, K_h, Q_h = (
            np.sin(k_e * self.core_width),
            np.sin(q_e * self.shell_width),
            np.sin(k_h * self.core_width),
            np.sin(q_h * self.shell_width),
        )
        R, H = self.core_width, self.shell_width
        core_denom = K_e * K_h * 2 * (k_h * k_h - k_e * k_e)
        shell_denom = Q_e * Q_h * 2 * (q_h * q_h - q_e * q_e)
        if abs(core_denom) < 1e-4 or abs(shell_denom) < 1e-4:
            print("Yer a breakpoint, 'arry.")
        # The accompanying formula for these are in a Maxima file.
        # QDWavefunctionsAndIntegrals.wxmx
        core_integral = -(
            (k_h - k_e) * np.sin(R * (k_h + k_e))
            - (k_h + k_e) * np.sin(R * (k_h - k_e))
        ) / core_denom
        shell_integral = -(
            (q_h - q_e) * np.sin(H * (q_h + q_e))
            - (q_h + q_e) * np.sin(H * (q_h - q_e))
        ) / shell_denom

        return abs(core_integral + shell_integral) ** 2

    def numerical_overlap_integral(self):
        k_e, q_e, k_h, q_h = self.calculate_wavenumbers() * n_

        def ewf(x):
            return wavefunction(x, k_e, q_e, self.core_width / n_, self.shell_width / n_)

        def hwf(x):
            return wavefunction(x, k_h, q_h, self.core_width / n_, self.shell_width / n_)

        def overlap_integrand_real(x):
            return np.real(x * x * ewf(x) * hwf(x))

        def overlap_integrand_imag(x):
            return np.imag(x * x * ewf(x) * hwf(x))

        overlap_integral_real = quad(overlap_integrand_real, 0, self.radius / n_)
        overlap_integral_imag = quad(overlap_integrand_imag, 0, self.radius / n_)
        return abs(overlap_integral_real[0] + 1j * overlap_integral_imag[0]) ** 2 * n_** 2

    def print_e_wf_at_zero(self):
        """Prints the wavefunction at 0."""
        print(
            _wavefunction(
                0, self.calculate_wavenumbers()[0], self.core_width
            )
        )


    # TODO: Implement branch for eh/he coreshells.
    def localization_electron_min_width(self, shell_width: float = None):
        """Minimum core width for localization of electron for a given shell width."""
        if shell_width is None:
            shell_width = self.shell_width

        m = self.cmat.m_e / self.smat.m_e

        # This could use a cached value. This does not change.
        x1 = brentq(
            _x_residual_function, 0, np.pi - 1e-10, args=(self.cmat.m_e, self.smat.m_e)
        )

        # Same for this.
        k1 = (2 * self.cmat.m_e * m_e * self.ue * e) ** 0.5 / hbar

        def min_core_loc_from_shell(r: float) -> float:
            return shell_width + m * r / (1 - m + 1 / _tanxdivx(k1 * r))
        if type(x1) == float:
            print('x1:', x1, 'k1:', k1)
            print('Low:', x1 / k1)
            print('High:', np.pi / k1)
            print('FLow:', min_core_loc_from_shell(x1 / k1))
            print('FLow+:', min_core_loc_from_shell(x1 / k1 + 1e-7 * n_))
            print('FHigh:', min_core_loc_from_shell(np.pi / k1))
            print('FHigh-:', min_core_loc_from_shell(np.pi / k1 - 1e-7 * n_))
            # result = brentq(min_core_loc_from_shell, x1 / k1 + 1e-10, np.pi / k1 - 1e-10)
            return 1# result
        else:
            raise ValueError

    # TODO: Implement branch for eh/he coreshells.
    def localization_hole_min_radius(self, core_width: float = None):
        if core_width is None:
            core_width = self.core_width
        """Minimum core width for localization of electron for a given shell width."""
        q1 = (2 * self.smat.m_h * m_e * self.uh * e) ** 0.5 / hbar

        # print(q1)

        def min_shell_loc_from_core(h: float) -> float:
            return core_width + np.tan(q1 * h) * q1

        # h = np.linspace(np.pi/ (2 * q1) + 0.1, np.pi / q1, 100)
        # plt.plot(h, min_shell_loc_from_core(h))
        # plt.show()
        result = brentq(min_shell_loc_from_core, np.pi / (2 * q1) + 1e-12, np.pi / q1)
        # print(min_shell_loc_from_core(np.pi / (2 * q1)), min_shell_loc_from_core(np.pi / q1))
        return result

    def coulomb_screening_energy(self, relative_tolerance: float = 1e-8):
        coulomb_screening_operator = make_coulomb_screening_operator(self)

        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()

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

        return dblquad(
            coulomb_integrand, 0, self.radius, 0, self.radius, epsrel=relative_tolerance
        )

    def interface_polarization_energy(self, relative_tolerance: float = 1e-8):
        interface_polarization_operator = make_interface_polarization_operator(self)

        k_e, q_e, k_h, q_h = self.calculate_wavenumbers()

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

        return dblquad(
            polarization_integrand,
            0,
            self.radius,
            0,
            self.radius,
            epsrel=relative_tolerance,
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
