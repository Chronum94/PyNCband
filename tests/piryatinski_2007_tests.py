import matplotlib.pyplot as plt
import numpy as np

from pyncband import Material, CoreShellParticle

mathigher = Material(1.0, 0.0, 0.1, 0.6, 4.0)
matlower = Material(1.0, -0.5, 0.1, 0.6, 6.0)

cseh = CoreShellParticle(matlower, mathigher, 1.0, 1.0)
cshe = CoreShellParticle(mathigher, matlower, 1.0, 1.0)
# ==================================================================
# def overlap_integral_eh(core_width, shell_width):
#     cseh.set_core_width(core_width)
#     cseh.set_shell_width(shell_width)
#     return cseh.numerical_overlap_integral()
#
# vectorized_overlap_integral_eh = np.vectorize(overlap_integral_eh)
#
# def overlap_integral_he(core_width, shell_width):
#     cshe.set_core_width(core_width)
#     cshe.set_shell_width(shell_width)
#     return cshe.numerical_overlap_integral()
#
# vectorized_overlap_integral_he = np.vectorize(overlap_integral_he)
# # ==================================================================
core_radii = np.linspace(0.2, 10, 10)
shell_radii = np.linspace(0.2, 10, 10)
#
# core_radii_mesh, shell_radii_mesh = np.meshgrid(core_radii, shell_radii)
# overlap_eh = vectorized_overlap_integral_eh(core_radii_mesh, shell_radii_mesh)
# overlap_he = vectorized_overlap_integral_he(core_radii_mesh, shell_radii_mesh)
# # ==================================================================
# colour_levels = np.linspace(0.0, 1.0, 10)
# plt.subplots()
# plt.subplot(121)
# plt.contourf(overlap_eh, extent=[0.2, 10, 0.2, 10], levels=colour_levels)
# plt.colorbar()
# plt.subplot(122)
# plt.contourf(overlap_he, extent=[0.2, 10, 0.2, 10], levels=colour_levels)
# plt.gcf().set_size_inches(20, 8)
# plt.colorbar()
# plt.show()
# # ===================================================================
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot_surface(core_radii_mesh, shell_radii_mesh, overlap_eh, ccount=20, rcount=20, cmap='inferno')
# ax.view_init(30, 45)
# plt.gcf().set_size_inches(10, 8)
# ax.set_xlabel('Core radius [nm]')
# ax.set_ylabel('Shell width [nm]')
# ax.set_zlabel('$\Theta$')
# plt.show()
# =====================================================================
def localization_in_core_eh(shell_width):
    return cseh.localization_electron_core(shell_width)

def localization_in_shell_eh(core_width):
    return cseh.localization_hole_shell(core_width)

vec_localization_in_core_eh = np.vectorize(localization_in_core_eh)
vec_localization_in_shell_eh = np.vectorize(localization_in_shell_eh)
# =====================================================================
core_radii_for_image = np.linspace(0.5, 4, 20)
shell_radii_for_image = np.linspace(0.2, 2, 20)
shell_radii_for_localization = vec_localization_in_shell_eh(core_radii_for_image)
shell_radii_asymptote = cseh.localization_hole_shell(10000)
core_radii_for_localization = vec_localization_in_core_eh(shell_radii_for_image)
core_radii_asymptote = cseh.localization_electron_core(10000)

plt.plot(core_radii_for_image, shell_radii_for_localization, core_radii_for_localization, shell_radii_for_image)
plt.xlim(0.5, 4)
plt.ylim(0.2, 2)
plt.show()