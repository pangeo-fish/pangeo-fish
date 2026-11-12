import numpy as np
import xarray as xr
import xdggs  # noqa: F401
from healpix_convolution.distances import _distances
from healpix_convolution.kernels.gaussian import gaussian_function


def _grid_info(grid):
    try:
        return grid.dggs.grid_info
    except ValueError as e:
        raise ValueError("invalid grid type") from e


def angular_distances(grid, pos):
    """Angular distance (in radians) from the cell containing ``pos`` to every cell.

    Parameters
    ----------
    grid : xarray.Dataset or xarray.DataArray
        HEALPix grid decoded with ``xdggs``.
    pos : mapping
        Position with ``longitude`` and ``latitude`` (in degrees).

    Returns
    -------
    distances : xarray.DataArray
        The great-circle distance between the center of the cell containing
        ``pos`` and the center of every cell of ``grid``, along ``cells``.
    """
    grid_info = _grid_info(grid)

    lon = pos["longitude"].data
    lat = pos["latitude"].data
    coord = grid.dggs.coord.variable
    cell_ids = coord.data  # type: np.ndarray
    center = np.reshape(grid_info.geographic2cell_ids(lon=lon, lat=lat), (1, 1))
    distances = _distances(
        center.astype(np.int64),
        np.reshape(cell_ids.astype(np.int64), (1, -1)),
        axis=-1,
        grid_info=grid_info,
    )

    return xr.DataArray(
        np.reshape(distances, (-1,)), dims="cells", coords={"cell_ids": coord}
    )


def normal_at(grid, pos, sigma):
    _grid_info(grid)

    if not pos:
        return None

    distances = angular_distances(grid, pos)
    pdf = gaussian_function(np.reshape(distances.data, (1, -1)), sigma)

    return distances.copy(data=np.reshape(pdf, (-1,)))
