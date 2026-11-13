"""Find the acoustic detections of an animal in the European Tracking Network (ETN) export.

ETN publishes its data as parquet files (https://www.lifewatch.be/etn/parquet/):

- ``acoustic/animals/part-0.parquet``: one row per animal and tag. Animals
  carrying a tag with both an acoustic transmitter and a data storage tag have
  ``tag_type == "acoustic-archival"``; ``acoustic_tag_id`` is the transmitter id.
- ``acoustic/detections/<animal_project_code>/part-0.parquet``: detections,
  partitioned by animal project.
- ``acoustic/deployments/part-0.parquet``: receiver deployments.
- ``acoustic/projects/part-0.parquet``: projects, with their moratorium flag.

:func:`find_etn_acoustic_detections` tells whether an animal (e.g. one carrying
a DST) also carries an acoustic tag and returns its detections and the receiver
deployments that detected it as the ``acoustic`` and ``stations`` nodes of a tag
:py:class:`xarray.DataTree` (see :func:`pangeo_fish.io.open_tag` and
``docs/format.md``).
"""

import warnings

import fsspec
import pandas as pd
import xarray as xr

# ETN_PARQUET_ROOT, _storage_options, _read_parquet and the de-duplication of
# the animals table (in _select_animal) are copies of those of
# pangeo_fish.etn_archival (PR #295): whichever of the two modules is merged
# second imports them from the first instead of keeping a copy.
ETN_PARQUET_ROOT = "https://www.lifewatch.be/etn/parquet"

ACOUSTIC_TAG_TYPES = ("acoustic", "acoustic-archival")

_STATION_COLUMNS = {
    "deployment_id": "deployment_id",
    "station_name": "station_name",
    "deploy_date_time": "deploy_time",
    "deploy_longitude": "deploy_longitude",
    "deploy_latitude": "deploy_latitude",
    "recover_date_time": "recover_time",
    "recover_longitude": "recover_longitude",
    "recover_latitude": "recover_latitude",
    "receiver_id": "receiver_id",
    "acoustic_project_code": "acoustic_project_code",
}


def _storage_options(path, storage_options):
    if storage_options is not None:
        return storage_options
    if str(path).startswith(("http://", "https://")):
        # honour the HTTP(S)_PROXY environment variables
        return {"client_kwargs": {"trust_env": True}}
    return {}


def _read_parquet(path, storage_options=None, **kwargs):
    # through fsspec, so that only the byte ranges needed are read from remote files
    with fsspec.open(path, "rb", **_storage_options(path, storage_options)) as f:
        return pd.read_parquet(f, **kwargs)


def _naive_utc(values):
    values = pd.to_datetime(values)
    if getattr(values.dt, "tz", None) is not None:
        values = values.dt.tz_convert(None)
    return values


def _select_animal(animals, project, tag_id):
    """All the tag rows of the animal carrying ``tag_id`` in ``project``."""
    # the ETN animals table has duplicated rows
    animals = animals.drop_duplicates(["animal_id", "tag_serial_number"])
    animals = animals[animals["animal_project_code"] == project]
    ids = animals.loc[animals["tag_serial_number"] == str(tag_id), "animal_id"]
    ids = ids.unique()
    if len(ids) == 0:
        raise ValueError(
            f"tag {tag_id!r} not found in the ETN animals table of project {project!r}"
        )
    if len(ids) > 1:
        raise ValueError(
            f"tag {tag_id!r} is attached to several animals of project {project!r}"
            f" ({sorted(ids.tolist())}): pass the animals table restricted to one"
            " of them as `animals`"
        )
    return animals[animals["animal_id"] == ids[0]]


def _read_stations(root, storage_options, detections):
    ids = detections["deployment_id"].dropna().unique().tolist()
    deployments = _read_parquet(
        f"{root}/acoustic/deployments/part-0.parquet",
        storage_options,
        columns=list(_STATION_COLUMNS),
        filters=[("deployment_id", "in", ids)],
    )
    deployments = deployments.rename(columns=_STATION_COLUMNS).drop_duplicates(
        "deployment_id"
    )
    for column in ["deploy_time", "recover_time"]:
        deployments[column] = _naive_utc(deployments[column])

    missing = sorted(set(ids) - set(deployments["deployment_id"]))
    if missing:
        warnings.warn(
            f"deployments {missing} are not in the ETN deployments table:"
            " using the positions of the detections",
            UserWarning,
            stacklevel=3,
        )
        fallback = (
            detections[detections["deployment_id"].isin(missing)]
            .drop_duplicates("deployment_id")
            .rename(
                columns={"longitude": "deploy_longitude", "latitude": "deploy_latitude"}
            )
        )
        deployments = pd.concat(
            [deployments, fallback.reindex(columns=deployments.columns)]
        )
    deployments = deployments.set_index("deployment_id").sort_index()

    # stations without deploy / recover time are never active in the acoustic
    # emission: use the first / last detection at the deployment instead
    by_deployment = detections.groupby("deployment_id")["time"]
    bounds = {"deploy_time": by_deployment.min(), "recover_time": by_deployment.max()}
    filled = {}
    for column, values in bounds.items():
        unknown = deployments[column].isna()
        if unknown.any():
            deployments.loc[unknown, column] = values.reindex(
                deployments.index[unknown]
            )
            filled[column] = deployments.index[unknown].tolist()
        deployments[column] = pd.to_datetime(deployments[column])
    if filled:
        details = "; ".join(f"{column}: {ids}" for column, ids in filled.items())
        warnings.warn(
            "deployments without deploy or recover time in ETN: using the first"
            f" or last detection at the deployment instead ({details})",
            UserWarning,
            stacklevel=3,
        )
    return deployments


def _moratorium(root, storage_options, project):
    try:
        projects = _read_parquet(
            f"{root}/acoustic/projects/part-0.parquet",
            storage_options,
            columns=["project_code", "moratorium"],
        )
    except (OSError, KeyError, ValueError):
        return None
    match = projects.loc[projects["project_code"] == project, "moratorium"]
    return None if match.empty else bool(match.iloc[0])


def find_etn_acoustic_detections(
    project,
    tag_id,
    *,
    root=ETN_PARQUET_ROOT,
    storage_options=None,
    animals=None,
):
    """Find the acoustic tags and detections of an ETN animal.

    The animal is identified like in
    :func:`pangeo_fish.etn_archival.open_etn_archival`: by its project and the
    serial number of any of its tags (e.g. its DST). A tag attached to several
    animals of ``project`` raises an error: pass ``animals`` restricted to the
    right animal.

    Deployments without deploy or recover time in ETN would never be active in
    the acoustic emission probability: their missing bounds are replaced by the
    first or last detection at that deployment, with a warning.

    .. note::
        The returned nodes can be added to a tag (e.g. from
        :func:`pangeo_fish.etn_archival.open_etn_archival`) and written with
        :func:`pangeo_fish.io.write_tag`. To keep the stations of an existing
        ``stations.csv``, merge them first with
        :func:`pangeo_fish.helpers.update_stations`.

    Parameters
    ----------
    project : str
        ETN animal project code.
    tag_id : str
        Serial number of any tag of the animal.
    root : str, default: ETN_PARQUET_ROOT
        Root of the ETN parquet export (or of a local copy with the same layout).
    storage_options : mapping, optional
        Passed to :py:func:`fsspec.open`.
    animals : pandas.DataFrame, optional
        Pre-loaded ETN animals table (``acoustic/animals/part-0.parquet``).

    Returns
    -------
    detections : xarray.DataTree
        Tree with the attributes ``etn_animal_id``, ``animal_project_code``,
        ``double_tagged`` (whether the animal carries both an acoustic and an
        archival tag) and ``acoustic_tag_ids`` (its transmitter ids, including
        the alternative ids of sensor tags). If detections are found, it has
        the nodes ``acoustic`` (``deployment_id``, ``longitude`` and
        ``latitude`` along the time zone naive, UTC ``time``) and ``stations``
        (receiver deployments along ``deployment_id``, with the columns of
        ``stations.csv``); otherwise it has no nodes and a warning explains why.
    """
    if animals is None:
        animals = _read_parquet(
            f"{root}/acoustic/animals/part-0.parquet", storage_options
        )
    rows = _select_animal(animals, project, tag_id)
    animal_id = int(rows["animal_id"].iloc[0])

    tag_types = set(rows["tag_type"].dropna())
    double_tagged = (
        "acoustic-archival" in tag_types
        or {
            "acoustic",
            "archival",
        }
        <= tag_types
    )
    # sensor tags may transmit under a second id (``acoustic_tag_id_alternative``)
    id_columns = [
        column
        for column in ["acoustic_tag_id", "acoustic_tag_id_alternative"]
        if column in rows.columns
    ]
    acoustic_rows = rows.loc[rows["tag_type"].isin(ACOUSTIC_TAG_TYPES), id_columns]
    acoustic_tag_ids = sorted(
        acoustic_rows.stack().dropna().astype(str).str.strip().unique().tolist()
    )
    acoustic_tag_ids = [tag_id for tag_id in acoustic_tag_ids if tag_id]

    attrs = {
        "etn_animal_id": animal_id,
        "animal_project_code": project,
        "double_tagged": double_tagged,
        "acoustic_tag_ids": acoustic_tag_ids,
    }

    def nothing_found(reason):
        warnings.warn(
            f"animal {animal_id} ({project}): {reason}", UserWarning, stacklevel=3
        )
        return xr.DataTree(xr.Dataset(attrs=attrs))

    if not acoustic_tag_ids:
        return nothing_found("no acoustic tag in ETN")

    source = f"{root}/acoustic/detections/{project}/part-0.parquet"
    try:
        raw = _read_parquet(
            source,
            storage_options,
            columns=[
                "date_time",
                "deployment_id",
                "deploy_longitude",
                "deploy_latitude",
                "station_name",
                "receiver_id",
                "animal_id",
            ],
            filters=[("animal_id", "==", animal_id)],
        )
    except FileNotFoundError:
        moratorium = _moratorium(root, storage_options, project)
        reason = {
            True: "the project is under moratorium",
            False: "the project has no published detections",
            None: "the project may be under moratorium or absent from the export",
        }[moratorium]
        return nothing_found(f"no detection file for project {project!r} ({reason})")

    if raw.empty:
        return nothing_found(
            f"acoustic tag(s) {acoustic_tag_ids} but 0 detections in ETN"
        )

    detections = (
        raw.rename(
            columns={
                "date_time": "time",
                "deploy_longitude": "longitude",
                "deploy_latitude": "latitude",
            }
        )
        .drop(columns="animal_id")
        .assign(time=lambda df: _naive_utc(df["time"]))
        .sort_values("time")
        .reset_index(drop=True)
    )
    stations = _read_stations(root, storage_options, detections)
    acoustic = detections.set_index("time")[["deployment_id", "longitude", "latitude"]]

    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs=attrs),
            "acoustic": acoustic.to_xarray(),
            "stations": stations.to_xarray(),
        }
    )
