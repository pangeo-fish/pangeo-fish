from math import atan2, cos, radians, sin, sqrt

import cf_xarray  # noqa: F401
import more_itertools
import xarray as xr
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)


def clear_attrs(obj, variables=None):
    # TODO: remove this after figuring out how to port this to upstream xarray
    new_obj = obj.copy()
    new_obj.attrs.clear()

    if variables is None:
        variables = []
    elif variables == "all":
        variables = list(getattr(new_obj, "variables", new_obj.coords))

    for name in more_itertools.always_iterable(variables):
        new_obj[name].attrs.clear()
    return new_obj


def postprocess_depth(ds):
    # TODO: remove this as it is unused
    new_names = {
        detected: standard
        for standard, (detected,) in ds.cf.standard_names.items()
        if detected in ds.coords
    }
    return ds.rename(new_names)


def normalize(obj, dim):
    return obj / obj.sum(dim=dim)


def _detect_dims(ds, guesses):
    for guess in guesses:
        try:
            coords = ds.cf[guess]
        except KeyError:
            continue

        return list(coords.dims)

    return None


def _detect_spatial_dims(
    ds, guesses=[["Y", "X"], ["latitude", "longitude"], ["x", "y"]]
):
    spatial_dims = _detect_dims(ds, guesses)
    if spatial_dims is None:
        raise ValueError(
            "could not determine spatial dimensions. Try"
            " calling `.cf.guess_coord_axis()` on the dataset."
        )

    return spatial_dims


def _detect_temporal_dims(ds, guesses=["T", "time"]):
    temporal_dims = _detect_dims(ds, guesses)
    if temporal_dims is None:
        raise ValueError(
            "could not determine temporal dimensions. Try"
            " calling `.cf.guess_coord_axis()` on the dataset."
        )

    return temporal_dims


def temporal_resolution(time):
    import pandas as pd
    from pandas.tseries.frequencies import to_offset

    freq = xr.infer_freq(time)
    timedelta = pd.Timedelta(to_offset(freq))
    units = timedelta.unit

    return xr.DataArray(
        timedelta.to_numpy().astype("float"), dims=None, attrs={"units": units}
    )


def progress_status(sequence):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.fields[label]}"),
        BarColumn(),
        TimeRemainingColumn(),
        transient=True,
    )

    with progress:
        task_id = progress.add_task(
            "computations", total=len(sequence), label="starting up"
        )
        for item in sequence:
            progress.update(task_id, advance=0, label=item)
            yield item
            progress.update(task_id, advance=1, label=item)


def haversine_distance(
    lon1: float, lon2: float, lat1: float, lat2: float, radius: float = 6371.0
):
    """
    Compute the Haversine distance between two points assuming the Earth is a perfect sphere.
    Implementation from https://stackoverflow.com/a/71412448.

    Parameters
    ----------
    - lon1 : float
        Longitude of the first point in degrees.
    - lat1 : float
        Latitude of the first point in degrees.
    - lon2 : float
        Longitude of the second point in degrees.
    - lat2 : float
        Latitude of the second point in degrees.
    - radius : float, default: 6731.0
        Radius of the Earth in kilometers.

    Returns
    -------
    float
        The Haversine distance in kilometers.
    """
    # source from: https://stackoverflow.com/a/71412448
    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius * c
