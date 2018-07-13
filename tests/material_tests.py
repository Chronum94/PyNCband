import numpy.testing as test

from PyNCband.Material import Material
from PyNCband.CoreShellParticle import CoreShellParticle

if __name__ == "__main__":
    AlSb = Material(1.6, -2.9, 0.12, 0.98, 2, "AlSb")
    CdS = Material(2.4, -3.5, 0.21, 0.8, 3, "CdS")
    test.assert_almost_equal(AlSb.vbe, -4.5)
    test.assert_almost_equal(CdS.vbe, -5.9)

    CdS_AlSb = CoreShellParticle(CdS, AlSb, 5, 3)
    AlSb_CdS = CoreShellParticle(AlSb, CdS, 5, 3)
    assert CdS_AlSb.type_two, "CdS/AlSb is is a type 2 QD"
    assert CdS_AlSb.e_h, "CdS/AlSb should be e-h core-shell."
    assert AlSb_CdS.type_two, "AlSb/CdS is is a type 2 QD"
    assert AlSb_CdS.h_e, "AlSb/CdS should be h-e core-shell."
    print(CdS_AlSb.ue)
    print(CdS_AlSb.calculate_wavenumbers())
    # print(CdS_AlSb.calculate_s1_energies())
    # s1_e_CdS_AlSb, s1_h_CdS_AlSb = CdS_AlSb.calculate_s1_energies()
    # print("S1 energy:", s1_e_CdS_AlSb)
    # ke, qe = CdS_AlSb.calculate_wavenumbers()
    # print("Electron wavevector in core and shell:", ke, qe)
    """ x, dx = np.linspace(1e-14, 10, 1000, retstep=True)
    wf =  CdS_AlSb.plot_electron_wavefunction(x, ke, qe)
    plt.plot(x, wf * wf)
    plt.show()
    print(CdS_AlSb.cmat.m_e, CdS_AlSb.smat.m_e)
    cme, sme = CdS_AlSb.cmat.m_e, CdS_AlSb.smat.m_e
    plt.plot(x, np.gradient(wf, dx) * np.where(x < CdS_AlSb.core_width, 1, 4))
    # plt.xlim(4.8, 5.2)
    plt.show() """

    CdS_AlSb.plot_potential_profile()
