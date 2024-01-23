import io
import json
import os

import datatree
import fsspec
import pandas as pd
import xarray as xr


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


def open_tag(root, name, storage_options=None):
    """open a tag

    Parameters
    ----------
    root : str or fsspec.FSMap
        The tag root. If not a mapper object, ``storage_options`` need
        to contain the necessary access options.
    name : str
        The DST name of the tag.
    storage_options : mapping, optional
        The storage options required to open the mapper. Only used if ``root`` is a url.

    Returns
    -------
    tag : datatree.DataTree
        The opened tag with involved stations, acoustic tags, tag log and metadata
    """
    if isinstance(root, str):
        if storage_options is None:
            storage_options = {}
        mapper = fsspec.get_mapper(root, **storage_options)
    else:
        mapper = root

    dst = pd.read_csv(mapper.dirfs.open(f"{name}/dst.csv"), index_col=0)
    dst_deployment = pd.read_csv(
        mapper.dirfs.open(f"{name}/dst_deployment.csv"), index_col=0
    )
    acoustic = pd.read_csv(mapper.dirfs.open(f"{name}/acoustic.csv"), index_col=0)
    metadata = json.load(mapper.dirfs.open(f"{name}/metadata.csv"))

    all_stations = pd.read_csv(
        mapper.dirfs.open("stations.csv"),
        parse_dates=["deploy_date_time", "recover_date_time"],
        date_format="%Y-%m-%d %H:%M:%S",
        index_col=0,
    )
    stations = all_stations.loc[acoustic["deployment_id"].unique()]

    return datatree.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs=metadata),
            "stations": stations.to_xarray(),
            "dst": dst.to_xarray(),
            "dst_deployment": dst_deployment.to_xarray(),
            "acoustic": acoustic.to_xarray(),
        }
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
    )

    return ds
