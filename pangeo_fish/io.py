import io
import os

import fsspec
import pandas as pd
import xarray as xr

from pangeo_fish.dataset_utils import broadcast_variables


def tz_convert(df, timezones):
    """Convert the timezone of columns in a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe.
    timezones : mapping of str to str
        The time zones to convert to per column.
    """
    new_columns = {
        column: pd.Index(df[column]).tz_convert(tz) for column, tz in timezones.items()
    }

    return df.assign(**new_columns)


def read_tag_database(url):
    """read the tag database

    Parameters
    ----------
    url : str, path-like or file-like
        The file path, url, path-like or pre-opened file object of the file to open

    Returns
    -------
    tag_database : pandas.DataFrame
        The opened tag database
    """
    return pd.read_csv(url, sep=";")


def read_detection_database(url):
    """read the detection database

    Parameters
    ----------
    url : str, path-like or file-like
        The file path, url, path-like or pre-opened file object of the file to open

    Returns
    -------
    detection_database : pandas.DataFrame
        The opened detection database

    """
    if isinstance(url, os.PathLike):
        url = os.fsspath(url)

    if isinstance(url, str):
        opened = fsspec.open(url, mode="r")
    else:
        opened = url

    with opened as f:
        # work around the weird quoting
        lines = (line.replace('"', "") for line in f)
        data = "\n".join(lines)

    content = io.StringIO(data)
    return (
        pd.read_csv(content, parse_dates=[1])
        .rename(columns={"date_time": "time"})
        .set_index("time")
    )


def open_copernicus_catalog(cat):
    """assemble the given intake catalog into a dataset

    .. warning::
        This will only work for the catalog at https://data-taos.ifremer.fr/references/copernicus.yaml

    Parameters
    ----------
    cat : intake.Catalog
        The pre-opened intake catalog

    Returns
    -------
    ds : xarray.Dataset
        The assembled dataset.
    """
    ds = (
        cat.data(type="TEM")
        .to_dask()
        .rename({"thetao": "TEMP"})
        .get(["TEMP"])
        .assign_coords({"time": lambda ds: ds["time"].astype("datetime64[ns]")})
        .assign(
            {
                "XE": cat.data(type="SSH").to_dask().get("zos"),
                "H0": (
                    cat.data_tmp(type="mdt")
                    .to_dask()
                    .get("deptho")
                    .rename({"latitude": "lat", "longitude": "lon"})
                ),
                "mask": (
                    cat.data_tmp(type="mdt")
                    .to_dask()
                    .get("mask")
                    .rename({"latitude": "lat", "longitude": "lon"})
                ),
            }
        )
        .assign(
            {
                "dynamic_depth": lambda ds: (ds["depth"] + ds["XE"]).assign_attrs(
                    {"units": "m", "positive": "down"}
                ),
                "dynamic_bathymetry": lambda ds: (ds["H0"] + ds["XE"]).assign_attrs(
                    {"units": "m", "positive": "down"}
                ),
            }
        )
        .pipe(broadcast_variables, {"lat": "latitude", "lon": "longitude"})
    )

    return ds
