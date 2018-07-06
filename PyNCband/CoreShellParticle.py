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

    def is_type_one(self):
        return False