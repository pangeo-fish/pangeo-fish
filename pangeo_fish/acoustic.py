import flox.xarray
import pandas as pd
import xarray as xr


def extract_receivers(
    detections,
    columns=[
        "deployment_id",
        "deploy_latitude",
        "deploy_longitude",
        "station_name",
    ],
):
    """extract the generic receiver information from the detection database

    Parameters
    ----------
    detections : pandas.DataFrame
        All detections in the database.
    columns : list of hashable, default: ["deployment_id", "deploy_latitude", \
                                          "deploy_longitude", "station_name"]
        Receiver-specific columns.

    Returns
    -------
    pandas.DataFrame
        The extracted receiver information.
    """
    subset = detections[columns]

    return subset.set_index("deployment_id").drop_duplicates()


def search_acoustic_tag_id(tag_database, pit_tag_id):
    """translate DST tag ID to the ID of the acoustic emitter

    Parameters
    ----------
    tag_database : pandas.DataFrame
        The database containing information about the individual deployments.
    pit_tag_id : str
        The ID of the DST tag

    Returns
    -------
    str
        The ID of the acoustic tag

    Raises
    ------
    ValueError
        if the given DST tag ID is not in the database.
    """
    try:
        info = tag_database.set_index("pit_tag_number").loc[pit_tag_id]
        return info["acoustic_tag_id"]
    except KeyError:
        raise ValueError(f"unknown tag id: {pit_tag_id}") from None


def count_detections(detections, by):
    """count the amount of detections by interval

    Parameters
    ----------
    detections : xarray.Dataset
        The detections for a specific tag
    by
        The values to group by. Can be anything that `flox` accepts.

    Returns
    -------
    count : xarray.Dataset
        The counts per interval

    See Also
    --------
    flox.xarray.xarray_reduce
    xarray.Dataset.groupby
    """
    count_on = (
        detections[["deployment_id"]]
        .assign(count=lambda ds: xr.ones_like(ds["deployment_id"], dtype=int))
        .set_coords(["deployment_id"])
    )

    isbin = [False, isinstance(by, pd.IntervalIndex)]

    result = flox.xarray.xarray_reduce(
        count_on,
        "deployment_id",
        "time",
        expected_groups=(None, by),
        isbin=isbin,
        func="sum",
        fill_value=0,
    )

    return result


def select_detections_by_tag_id(database, tag_id):
    """select detections by the acoustic tag id

    Parameters
    ----------
    database : pandas.DataFrame
        The detections database.
    tag_id : str
        The acoustic tag id to search for.

    Returns
    -------
    detections : xarray.Dataset
        The selected detections.
    """
    return (
        database[["deployment_id", "acoustic_tag_id"]]
        .to_xarray()
        .set_coords(["acoustic_tag_id"])
        .set_xindex("acoustic_tag_id")
        .sel({"acoustic_tag_id": tag_id})
        .drop_vars(["acoustic_tag_id"])
    )
