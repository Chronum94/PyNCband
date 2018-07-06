from .Material import Material


class CoreShellParticle:
    def __init__(
        self,
        core_material: Material,
        shell_material: Material,
        core_thickness: float,
        shell_thickness: float,
    ):
        self.cmat = core_material
        self.smat = shell_material
        self.core_width = core_thickness
        self.shell_width = shell_thickness

        self.type_one = self.is_type_one()
        self.type_two, self.he, self.eh = self.is_type_two()

    # This is likely to get refactored later to return types.
    def is_type_one(self):
        return (self.cmat.vbe > self.smat.vbe) and (self.cmat.cbe < self.smat.cbe)

    def is_type_two(self):
        core_higher = (self.cmat.vbe > self.smat.vbe) and (
            self.cmat.cbe > self.smat.cbe
        )
        shell_higher = (self.cmat.vbe < self.smat.vbe) and (
            self.cmat.cbe < self.smat.cbe
        )
        return core_higher or shell_higher, core_higher, shell_higher
