import numpy.testing as test

from PyNCband.Material import Material
from PyNCband.CoreShellParticle import CoreShellParticle

if __name__ == "__main__":
    AlSb = Material(1.6, -2.9, 0.12, 0.98, 2, "AlSb")
    CdS = Material(2.4, -3.5, 0.21, 0.8, 3, "CdS")
    test.assert_almost_equal(AlSb.vbe, -4.5)
    test.assert_almost_equal(CdS.vbe, -5.9)

    CdS_AlSb = CoreShellParticle(CdS, AlSb, 5, 5)
    AlSb_CdS = CoreShellParticle(AlSb, CdS, 5, 5)
    assert CdS_AlSb.type_two, "CdS/AlSb is is a type 2 QD"
    assert CdS_AlSb.eh, "CdS/AlSb should be e-h core-shell."
    assert AlSb_CdS.type_two, "AlSb/CdS is is a type 2 QD"
    assert AlSb_CdS.he, "AlSb/CdS should be h-e core-shell."
    assert CdS_AlSb.ue < 0
    assert CdS_AlSb.uh < 0
    print(CdS_AlSb.ue)
    print(CdS_AlSb.calculate_electron_wavevectors(0 ))
