
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad, dblquad

from .Material import Material
from .physicsfunctions import *

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
        self.core_width = core_width
        self.shell_width = shell_width
        self.radius = core_width + shell_width
        self.type_one = self.is_type_one()
        self.type_two, self.h_e, self.e_h = self.is_type_two()

        self.ue = np.abs(self.cmat.cbe - self.smat.cbe)
        self.uh = np.abs(self.cmat.vbe - self.smat.vbe)

    # This is likely to get refactored later to return types.
    def is_type_one(self):
        return (self.cmat.vbe > self.smat.vbe) and (self.cmat.cbe < self.smat.cbe)

    def is_type_two(self):
        """"A type two QD has both conduction and valence band edges of its core either higher or lower than the
        corresponding band edges of the shell."""
        core_higher = (self.cmat.vbe > self.smat.vbe) and (
            self.cmat.cbe > self.smat.cbe
        )
        shell_higher = (self.cmat.vbe < self.smat.vbe) and (
            self.cmat.cbe < self.smat.cbe
        )
        return core_higher or shell_higher, core_higher, shell_higher

    def calculate_wavevectors(self):
        """Returns a tuple of the electron wavevectors in the core and the shell."""
        energy_e, energy_h = self.calculate_s1_energies()
        if self.e_h:
            return (
                wavevector_from_energy(energy_e, self.cmat.m_e),
                wavevector_from_energy(
                    energy_e, self.smat.m_e, potential_offset=self.ue
                ),
                wavevector_from_energy(
                    energy_h, self.cmat.m_h, potential_offset=self.uh
                ),
                wavevector_from_energy(energy_h, self.smat.m_h),
            )
        elif self.h_e:
            return (
                wavevector_from_energy(
                    energy_e, self.cmat.m_e, potential_offset=self.ue
                ),
                wavevector_from_energy(energy_e, self.smat.m_e),
                wavevector_from_energy(energy_h, self.cmat.m_h),
                wavevector_from_energy(
                    energy_h, self.smat.m_h, potential_offset=self.uh
                ),
            )

    def calculate_s1_energies(self, bounds=(), resolution=1000):
        lower_bound_e = self.ue + 1e-14
        upper_bound_e = 10 * self.ue
        lower_bound_h = self.uh + 1e-14
        upper_bound_h = 10 * self.uh

        x = np.linspace(lower_bound_e, upper_bound_e, resolution)
        if bounds != ():
            x = np.linspace(bounds[0], bounds[1], resolution)
        ye = electron_eigenvalue_residual(x, self)
        # print(np.all(np.isreal(yh)))

        ye_signs = np.sign(ye)
        ye_sign_change = np.diff(ye_signs)  # This array is one element shorter.
        ye_neg2pos_change = np.argwhere(np.where(ye_sign_change > 0.5, 1, 0))
        root_position = ye_neg2pos_change[0]
        # print(*x[root_position:root_position + 2])
        s1_electron_energy = brentq(
            electron_eigenvalue_residual,
            x[root_position],
            x[root_position + 1],
            args=self,
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

        s1_hole_energy = brentq(
            hole_eigenvalue_residual,
            x[root_position],
            x[root_position + 1],
            args=self,
        )
        # print(s1_electron_energy)
        # plt.plot(x, yh)
        # plt.ylim(-10, 10)
        # plt.show()
        return s1_electron_energy, s1_hole_energy

    def plot_electron_wavefunction(
        self, x, core_wavevector: float, shell_wavevector: float
    ):

        # x = np.linspace(1e-10, self.core_width + self.shell_width, 1000)
        # cwf = lambda x: unnormalized_core_wavefunction(x, core_wavevector, self.core_width)
        # swf = lambda x: unnormalized_shell_wavefunction(
        #     x, shell_wavevector, self.core_width, self.shell_width
        # )
        #
        # y = np.piecewise(
        #     x,
        #     [
        #         x < self.core_width,
        #         x > self.core_width,
        #         x > self.core_width + self.shell_width,
        #     ],
        #     [cwf, swf, 0],
        # )
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
        k_e, q_e, k_h, q_h = self.calculate_wavevectors()
        K_e, Q_e, K_h, Q_h = (
            np.sin(k_e * self.core_width),
            np.sin(q_e * self.shell_width),
            np.sin(k_h * self.core_width),
            np.sin(q_h * self.shell_width),
        )
        R, H = self.core_width, self.shell_width

        # The accompanying formula for these are in a Maxima file.
        # QDWavefunctionsAndIntegrals.wxmx
        core_integral = -(
            (k_h - k_e) * np.sin(R * (k_h + k_e))
            - (k_h + k_e) * np.sin(R * (k_h - k_e))
        ) / (K_e * K_h * 2 * (k_h * k_h - k_e * k_e))
        shell_integral = -(
            (q_h - q_e) * np.sin(H * (q_h + q_e))
            - (q_h + q_e) * np.sin(H * (q_h - q_e))
        ) / (Q_e * Q_h * 2 * (q_h * q_h - q_e * q_e))

        return abs(core_integral + shell_integral) ** 2

    def numerical_overlap_integral(self):
        k_e, q_e, k_h, q_h = self.calculate_wavevectors()

        def ewf(x): return wavefunction(x, k_e, q_e, self.core_width, self.shell_width)

        def hwf(x): return wavefunction(x, k_h, q_h, self.core_width, self.shell_width)

        def overlap_integrand(x): return x * x * ewf(x) * hwf(x)

        overlap_integral = quad(overlap_integrand, 0, self.radius)
        return abs(overlap_integral[0]) ** 2

    def print_e_wf_at_zero(self):
        print(
            unnormalized_core_wavefunction(
                1e-14, self.calculate_wavevectors()[0], self.core_width
            )
        )

    def __normalize_wavefunction(self):
        raise NotImplementedError

    def localization_electron_min_width(self, shell_width: float = None):
        if shell_width is None:
            shell_width = self.shell_width
        """Minimum core width for localization of electron for a given shell width."""
        m = self.cmat.m_e / self.smat.m_e
        x1 = brentq(
            _x_residual_function, 0, np.pi / 1.001, args=(self.cmat.m_e, self.smat.m_e)
        )
        k1 = (2 * self.cmat.m_e * self.ue) ** 0.5  # No 1/hbar because unitless.

        def min_core_loc_from_shell(r): return shell_width - m * r / (
            1 - m + k1 * r / np.tan(k1 * r)
        )
        result = brentq(min_core_loc_from_shell, x1 / k1, np.pi / k1)
        return result

    def localization_hole_min_radius(self, core_width: float = None):
        if core_width is None:
            core_width = self.core_width
        """Minimum core width for localization of electron for a given shell width."""
        q1 = (2 * self.smat.m_h * self.uh) ** 0.5  # No 1/hbar because unitless.
        print(q1)

        def min_shell_loc_from_core(h): return core_width + np.tan(q1 * h) * q1
        # h = np.linspace(np.pi/ (2 * q1) + 0.1, np.pi / q1, 100)
        # plt.plot(h, min_shell_loc_from_core(h))
        # plt.show()
        result = brentq(min_shell_loc_from_core, np.pi / (2 * q1) + 1e-12, np.pi / q1)
        # print(min_shell_loc_from_core(np.pi / (2 * q1)), min_shell_loc_from_core(np.pi / q1))
        return result

    def coulomb_screening_energy(self):
        coulomb_screening_operator = make_coulomb_screening_operator(self)

        k_e, q_e, k_h, q_h = self.calculate_wavevectors()

        # Electron/hole density functions.
        edf = (
            lambda x: abs(wavefunction(x, k_e, q_e, self.core_width, self.shell_width))
            ** 2
        )
        hdf = (
            lambda x: abs(wavefunction(x, k_h, q_h, self.core_width, self.shell_width))
            ** 2
        )

        coulomb_integrand = (
            lambda r1, r2: r1 ** 2
            * r2 ** 2
            * edf(r1)
            * hdf(r2)
            * coulomb_screening_operator(r1, r2)
        )

        coulomb_integral = dblquad(coulomb_integrand, 0, self.radius, 0, self.radius)
        return coulomb_integral

    def interface_polarization_energy(self):
        interface_polarization_operator = make_interface_polarization_operator(self)

        k_e, q_e, k_h, q_h = self.calculate_wavevectors()

        # Electron/hole density functions.
        edf = (
            lambda x: abs(wavefunction(x, k_e, q_e, self.core_width, self.shell_width))
            ** 2
        )
        hdf = (
            lambda x: abs(wavefunction(x, k_h, q_h, self.core_width, self.shell_width))
            ** 2
        )

        polarization_integrand = (
            lambda r1, r2: r1 ** 2
            * r2 ** 2
            * edf(r1)
            * hdf(r2)
            * interface_polarization_operator(r1, r2)
        )

        polarization_integral = dblquad(polarization_integrand, 0, self.radius, 0, self.radius)
        return polarization_integral
