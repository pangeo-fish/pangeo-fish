import dask
import healpy as hp
import numpy as np
import xarray as xr


def geographic_to_astronomic(lat, lon, rot):
    """transform geographic coordinates to astronomic coordinates

    Parameters
    ----------
    lat : array-like
        geographic latitude, in degrees
    lon : array-like
        geographic longitude, in degrees
    rot : list-like
        Two element list with the rotation transformation (shift?) used by the grid, in
        degrees

    Returns
    -------
    theta : array-like
        Colatitude in degrees
    phi : array-like
        Astronomic longitude in degrees
    """
    theta = 90.0 - lat - rot[1]
    phi = -lon + rot[0]

    return theta, phi


def astronomic_to_cartesian(theta, phi, dim="receiver_id"):
    """transform astronomic coordinates to cartesian coordinates

    Parameters
    ----------
    theta : DataArray
        astronomic colatitude, in degrees
    phi : DataArray
        astronomic longitude, in degrees
    dim : hashable
        Name of the dimension

    Returns
    -------
    cartesian : Dataset
        Cartesian coordinates

    See Also
    --------
    healpy.ang2vec
    """
    # TODO: try to determine `dim` automatically
    cartesian = xr.apply_ufunc(
        hp.ang2vec,
        np.deg2rad(theta),
        np.deg2rad(phi),
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim, "cartesian"]],
    )

    return cartesian.assign_coords(cartesian=["x", "y", "z"])


def astronomic_to_cell_ids(nside, phi, theta):
    """Compute cell ids from astronomic coordinates

    Parameters
    ----------
    nside : int
        Healpix resolution level
    phi, theta : xr.DataArray
        astronomic longitude and colatitude, in degrees

    Returns
    -------
    cell_ids : xr.DataArray
        The computed cell ids
    """
    phi_, theta_ = dask.compute(phi, theta)

    cell_ids = xr.apply_ufunc(
        hp.ang2pix,
        nside,
        np.deg2rad(theta_),
        np.deg2rad(phi_),
        kwargs={"nest": True},
        input_core_dims=[[], ["x", "y"], ["x", "y"]],
        output_core_dims=[["x", "y"]],
    )

    return cell_ids


def buffer_points(
    cell_ids,
    positions,
    *,
    buffer_size,
    nside,
    sphere_radius=6371e3,
    factor=4,
    intersect=False,
):
    """select the cells within a circular buffer around the given positions

    Parameters
    ----------
    cell_ids : xarray.DataArray
        The cell ids within the given grid.
    positions : xarray.DataArray
        The positions of the points in cartesian coordinates.
    buffer_size : float
        The size of the circular buffer.
    nside : int
        The resolution of the healpix grid.
    sphere_radius : float, default: 6371000
        The radius of the underlying sphere, used to convert `radius` to radians. By
        default, this is the standard earth's radius in meters.
    factor : int, default: 4
        The increased resolution for the buffer search.
    intersect : bool, default: False
        If `False`, select all cells where the center is within the buffer. If `True`,
        select cells which intersect the buffer.

    Returns
    -------
    masks : xarray.DataArray
        The masks for each position. The cells within the buffer are `True`, every other
        cell is set to `False`.

    See Also
    --------
    pangeo_fish.healpy.geographic_to_astronomic
    pangeo_fish.healpy.astronomic_to_cartesian
    """

    def _buffer_masks(cell_ids, vector, nside, radius, factor=4, intersect=False):
        selected_cells = hp.query_disc(
            nside, vector, radius, nest=True, fact=factor, inclusive=intersect
        )
        return np.isin(cell_ids, selected_cells, assume_unique=True)

    radius_ = buffer_size / sphere_radius

    masks = xr.apply_ufunc(
        _buffer_masks,
        cell_ids,
        positions,
        input_core_dims=[["y", "x"], ["cartesian"]],
        kwargs={
            "radius": radius_,
            "nside": nside,
            "factor": factor,
            "intersect": intersect,
        },
        output_core_dims=[["y", "x"]],
        vectorize=True,
    )

    return masks.assign_coords(cell_ids=cell_ids)
