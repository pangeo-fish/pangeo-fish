import re

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
    # FIXME: remove this after figuring out how to port this to upstream xarray
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


units_re = re.compile(r"timedelta64\[(?P<units>.+?)\]")


def timedelta_units(arr):
    dtype = arr.dtype

    if dtype.kind != "m":
        raise ValueError("not a timedelta64")

    match = units_re.fullmatch(dtype.name)
    if match is None:
        raise ValueError("timedelta64 without units")

    return match.group("units")


def temporal_resolution(time):
    import pandas as pd
    from pandas.tseries.frequencies import to_offset

    freq = xr.infer_freq(time)
    timedelta = pd.Timedelta(to_offset(freq)).to_numpy()
    units = timedelta_units(timedelta)

    return xr.DataArray(timedelta.astype("float"), dims=None, attrs={"units": units})


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
