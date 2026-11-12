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
deployments that detected it; :meth:`ETNAcousticDetections.write` writes them as
the ``acoustic.csv`` and ``stations.csv`` files of the tag folder format (see
``docs/format.md``).
"""

import dataclasses
import os
import warnings

import fsspec
import pandas as pd

ETN_PARQUET_ROOT = "https://www.lifewatch.be/etn/parquet"

ACOUSTIC_TAG_TYPES = ("acoustic", "acoustic-archival")

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

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


@dataclasses.dataclass
class ETNAcousticDetections:
    """Acoustic information of an ETN animal.

    Attributes
    ----------
    animal_id : int
        ETN animal id.
    animal_project_code : str
        ETN animal project.
    double_tagged : bool
        Whether the animal carries both an acoustic and an archival tag.
    acoustic_tag_ids : list of str
        Acoustic transmitter ids of the animal, including the alternative ids
        of sensor tags (empty if none).
    detections : pandas.DataFrame
        Detections with the columns ``time`` (time zone naive, UTC),
        ``deployment_id``, ``longitude``, ``latitude``, ``station_name``,
        ``acoustic_tag_id`` and ``receiver_id``.
    stations : pandas.DataFrame
        The receiver deployments of the detections, indexed by ``deployment_id``,
        with the columns of ``stations.csv``.
    message : str
        Human readable summary, explaining why nothing was found if applicable.
    """

    animal_id: int
    animal_project_code: str
    double_tagged: bool
    acoustic_tag_ids: list
    detections: pd.DataFrame
    stations: pd.DataFrame
    message: str

    @property
    def found(self):
        """Whether any detection was found."""
        return len(self.detections) > 0

    def write(self, root, name):
        """Write the detections and stations to a tag folder.

        Writes ``{root}/{name}/acoustic.csv`` and ``{root}/stations.csv``.
        Deployments already present in an existing ``stations.csv`` are kept
        (the new rows replace those with the same ``deployment_id``).

        Parameters
        ----------
        root : str or os.PathLike
            Local root folder of the tags.
        name : str
            Name of the tag folder (usually the DST id).

        Returns
        -------
        paths : list of str
            The files written.
        """
        root = os.fspath(root)
        folder = os.path.join(root, name)
        os.makedirs(folder, exist_ok=True)

        acoustic = self.detections[["time", "deployment_id", "longitude", "latitude"]]
        acoustic = acoustic.assign(time=acoustic["time"].dt.tz_localize("UTC"))
        acoustic_path = os.path.join(folder, "acoustic.csv")
        acoustic.to_csv(
            acoustic_path, index=False, date_format=_TIME_FORMAT, na_rep="NA"
        )

        stations = self.stations
        stations_path = os.path.join(root, "stations.csv")
        if os.path.exists(stations_path):
            existing = pd.read_csv(
                stations_path,
                index_col="deployment_id",
                parse_dates=["deploy_time", "recover_time"],
            )
            for column in ["deploy_time", "recover_time"]:
                existing[column] = _naive_utc(existing[column])
            existing = existing[~existing.index.isin(stations.index)]
            stations = pd.concat([existing, stations]).sort_index()
        stations = stations.assign(
            **{
                column: stations[column].dt.tz_localize("UTC")
                for column in ["deploy_time", "recover_time"]
            }
        )
        stations.to_csv(stations_path, date_format=_TIME_FORMAT, na_rep="NA")

        return [acoustic_path, stations_path]


def _select_animal(animals, animal_id, tag_serial_number, project):
    if (animal_id is None) == (tag_serial_number is None):
        raise ValueError("pass exactly one of animal_id or tag_serial_number")
    if project is not None:
        animals = animals[animals["animal_project_code"] == project]
    if animal_id is not None:
        rows = animals[animals["animal_id"] == int(animal_id)]
        what = f"animal {animal_id!r}"
    else:
        ids = animals.loc[
            animals["tag_serial_number"] == str(tag_serial_number), "animal_id"
        ].unique()
        if len(ids) > 1:
            raise ValueError(
                f"tag {tag_serial_number!r} is attached to several animals"
                f" ({sorted(ids.tolist())}): pass animal_id or project"
            )
        rows = animals[animals["animal_id"].isin(ids)]
        what = f"tag {tag_serial_number!r}"
    if rows.empty:
        raise ValueError(f"{what} not found in the ETN animals table")
    return rows


def _empty_stations():
    return pd.DataFrame(columns=list(_STATION_COLUMNS.values())).set_index(
        "deployment_id"
    )


def _read_stations(root, storage_options, detections):
    ids = detections["deployment_id"].dropna().unique().tolist()
    if not ids:
        return _empty_stations()
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
            " using the positions of the detections, without deploy/recover times",
            UserWarning,
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
    if deployments["recover_time"].isna().any():
        warnings.warn(
            "some deployments have no recover time in ETN", UserWarning, stacklevel=3
        )
    return deployments.set_index("deployment_id").sort_index()


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
    animal_id=None,
    tag_serial_number=None,
    *,
    project=None,
    root=ETN_PARQUET_ROOT,
    storage_options=None,
    animals=None,
):
    """Find the acoustic tags and detections of an ETN animal.

    Parameters
    ----------
    animal_id : int, optional
        ETN animal id.
    tag_serial_number : str, optional
        Serial number of any tag of the animal (e.g. its DST). Exactly one of
        ``animal_id`` and ``tag_serial_number`` must be given.
    project : str, optional
        ETN animal project code, to disambiguate reused tag serial numbers.
    root : str, default: ETN_PARQUET_ROOT
        Root of the ETN parquet export (or of a local copy with the same layout).
    storage_options : mapping, optional
        Passed to :py:func:`fsspec.open`.
    animals : pandas.DataFrame, optional
        Pre-loaded ETN animals table (``acoustic/animals/part-0.parquet``).

    Returns
    -------
    result : ETNAcousticDetections
        ``result.found`` tells whether detections were found and
        ``result.message`` explains the result.
    """
    if animals is None:
        animals = _read_parquet(
            f"{root}/acoustic/animals/part-0.parquet", storage_options
        )
    # the ETN animals table has duplicated rows
    rows = _select_animal(
        animals.drop_duplicates(["animal_id", "tag_serial_number"]),
        animal_id,
        tag_serial_number,
        project,
    )
    animal_id = int(rows["animal_id"].iloc[0])
    project = str(rows["animal_project_code"].iloc[0])

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

    empty = pd.DataFrame(
        {
            "time": pd.Series(dtype="datetime64[ns]"),
            "deployment_id": pd.Series(dtype="int64"),
            "longitude": pd.Series(dtype=float),
            "latitude": pd.Series(dtype=float),
            "station_name": pd.Series(dtype=object),
            "acoustic_tag_id": pd.Series(dtype=object),
            "receiver_id": pd.Series(dtype=object),
        }
    )

    def result(detections, stations, message):
        return ETNAcousticDetections(
            animal_id=animal_id,
            animal_project_code=project,
            double_tagged=double_tagged,
            acoustic_tag_ids=acoustic_tag_ids,
            detections=detections,
            stations=stations,
            message=message,
        )

    who = f"animal {animal_id} ({project})"
    if not acoustic_tag_ids:
        return result(empty, _empty_stations(), f"{who} has no acoustic tag in ETN")

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
                "acoustic_tag_id",
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
        return result(
            empty,
            _empty_stations(),
            f"{who}: no detection file for project {project!r} ({reason})",
        )

    if raw.empty:
        return result(
            empty,
            _empty_stations(),
            f"{who} has acoustic tag(s) {acoustic_tag_ids} but 0 detections in ETN",
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
    message = (
        f"{who}: {len(detections)} detections"
        f" (transmitter ids {sorted(detections['acoustic_tag_id'].dropna().unique())})"
        f" at {detections['deployment_id'].nunique()} deployments"
    )
    return result(detections, stations, message)
