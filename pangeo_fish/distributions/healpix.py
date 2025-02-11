import numpy as np
import xarray as xr
import xdggs  # noqa: F401
from healpix_convolution.distances import _distances
from healpix_convolution.kernels.gaussian import gaussian_function


def normal_at(grid, pos, sigma):
    try:
        grid_info = grid.dggs.grid_info
    except ValueError as e:
        raise ValueError("invalid grid type") from e

    if not pos:
        return None

    lon = pos["longitude"].data
    lat = pos["latitude"].data
    coord = grid.dggs.coord.variable
    cell_ids = coord.data  # type: np.ndarray
    center = np.reshape(grid_info.geographic2cell_ids(lon=lon, lat=lat), (1, 1))
    distances = _distances(
        center.astype(np.int64),
        np.reshape(cell_ids.astype(np.int64), (1, -1)),
        axis=-1,
        nside=grid_info.nside,
        nest=grid_info.nest,
    )

    pdf = gaussian_function(distances, sigma)

    return xr.DataArray(np.squeeze(pdf), dims="cells", coords={"cell_ids": coord})
