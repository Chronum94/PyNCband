from multiprocessing import Pool
from typing import Iterable, Union

from .Material import Material
from .CoreShellParticle import CoreShellParticle

__all__ = ["coreshell_scan_heatmap"]


def coreshell_scan_heatmap(
    core_material: Material,
    shell_material: Material,
    core_limits: Iterable[int] = (1, 2),
    shell_limits: Iterable[int] = (1, 2),
    resolution: Union[int, Iterable[int]] = 25,
):
    """Conducts a parameter scan over the core and shell radii for a dot of given materials. The `resolution` parameter
    can be either an integer, in which case both parameters are scanned with the same resolution, or a 2-tuple of
    integers, where the core and shell radii are scanned according to the first and second resolution, respectively.

    Parameters
    ----------
    core_material
    shell_material
    core_limits
    shell_limits
    resolution

    Returns
    -------

    """
    raise NotImplementedError
