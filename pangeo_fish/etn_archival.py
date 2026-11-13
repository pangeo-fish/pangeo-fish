"""Read archival (DST) data from the public parquet export of the European Tracking Network (ETN).

ETN publishes its data as parquet files (https://www.lifewatch.be/etn/parquet/):

- ``archival/archival_det/<project>/part-0.parquet``: long table of sensor records
  (``tag_id``, ``timestamp_utc``, ``measurement_type``, ``measurement_value``,
  ``measurement_unit``), with row groups sorted by tag
- ``acoustic/animals/part-0.parquet``: animal metadata, including release and
  recapture information (shared by the acoustic and archival projects)
- ``archival/tags/part-0.parquet``: tag model and sensor ranges

:func:`open_etn_archival` converts one tag to the tag :py:class:`xarray.DataTree`
returned by :func:`pangeo_fish.io.open_tag`; :func:`pangeo_fish.io.write_tag`
writes such a tree to the tag folder format (see ``docs/format.md``).
"""

import warnings

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

ETN_PARQUET_ROOT = "https://www.lifewatch.be/etn/parquet"

#: ETN uses several spellings for the same sensor (e.g. "Pressure" and "pressure")
_MEASUREMENT_NAMES = {
    "pressure": "pressure",
    "depth": "pressure",
    "temperature": "temperature",
}

# ``measurement_unit`` of the pressure records -> ``pressure_unit`` metadata.
# ETN labels the DST pressure ``Pa`` but stores dbar (~ metres of depth).
_PRESSURE_UNITS = {
    "Pa": "dbar (labelled 'Pa' in ETN), used as depth in metres",
    "dbar": "dbar, used as depth in metres",
    "m": "m (depth)",
}
# deepest ocean ~ 11000 dbar: larger values labelled ``Pa`` are real pascals
_MAX_DBAR = 12000


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


def _archival_data_url(project, root=ETN_PARQUET_ROOT):
    """Path of the sensor records of an ETN archival project."""
    return f"{root}/archival/archival_det/{project}/part-0.parquet"


def read_etn_animals(root=ETN_PARQUET_ROOT, storage_options=None):
    """Read the ETN animals table, without its duplicated rows.

    Parameters
    ----------
    root : str, default: ETN_PARQUET_ROOT
        Root of the ETN parquet export (or of a local copy with the same layout).
    storage_options : mapping, optional
        Passed to :py:func:`fsspec.open`.

    Returns
    -------
    animals : pandas.DataFrame
        One row per animal and tag.
    """
    animals = _read_parquet(f"{root}/acoustic/animals/part-0.parquet", storage_options)
    return animals.drop_duplicates(["animal_id", "tag_serial_number"]).reset_index(
        drop=True
    )


def _read_sensor_ranges(tag_id, root=ETN_PARQUET_ROOT, storage_options=None):
    """Sensor ranges of an archival tag: ``{"temperature": [min, max], ...}``."""
    tags = _read_parquet(
        f"{root}/archival/tags/part-0.parquet",
        storage_options,
        columns=[
            "tag_serial_number",
            "sensor_type",
            "sensor_range_min",
            "sensor_range_max",
        ],
    )
    tags = tags[tags["tag_serial_number"] == tag_id]
    ranges = {}
    for row in tags.itertuples():
        sensor = _MEASUREMENT_NAMES.get(str(row.sensor_type).lower())
        bounds = pd.to_numeric(
            pd.Series([row.sensor_range_min, row.sensor_range_max]), errors="coerce"
        )
        if sensor is not None and bounds.notna().all():
            ranges[sensor] = [float(b) for b in bounds]
    return ranges


def _read_records(project, tag_id, root=ETN_PARQUET_ROOT, storage_options=None):
    """Read the sensor records of one tag as a ``time, temperature, pressure`` table.

    Exact duplicate records are removed. Records with the same time and sensor
    but different values raise an error.

    Parameters
    ----------
    project : str
        ETN project code (e.g. ``"shad_scheldt_dst"``).
    tag_id : str
        Tag serial number (e.g. ``"A19845"``).
    root : str, default: ETN_PARQUET_ROOT
        Root of the ETN parquet export (or of a local copy with the same layout).
    storage_options : mapping, optional
        Passed to :py:func:`fsspec.open`.

    Returns
    -------
    dst : pandas.DataFrame
        Indexed by (time zone naive, UTC) ``time``, with the columns
        ``temperature`` and ``pressure``. The values are returned unchanged.
    stats : dict
        Number of records read, units as labelled by ETN, duplicates removed,
        and ``pressure_unit``: the unit of ``pressure``, derived from the ETN
        label. ETN labels the DST pressure ``Pa`` but stores dbar (~ metres of
        depth); a warning is emitted if values labelled ``Pa`` are too large to
        be dbar, or if the label is not a known unit.
    """
    source = _archival_data_url(project, root)
    # the files are sorted by tag: the filter only reads the row groups of `tag_id`
    records = _read_parquet(
        source,
        storage_options,
        columns=[
            "tag_id",
            "timestamp_utc",
            "measurement_type",
            "measurement_value",
            "measurement_unit",
        ],
        filters=[("tag_id", "==", tag_id)],
    )
    if records.empty:
        raise ValueError(f"no sensor records for tag {tag_id!r} in {source}")

    units = (
        records[["measurement_type", "measurement_unit"]]
        .drop_duplicates()
        .astype(str)
        .agg(": ".join, axis=1)
    )
    stats = {"etn_records": len(records), "etn_units": sorted(units)}

    records["measurement_type"] = (
        records["measurement_type"].str.lower().map(_MEASUREMENT_NAMES)
    )
    records = records.dropna(subset=["measurement_type"])

    key = ["timestamp_utc", "measurement_type"]
    n_records = len(records)
    records = records.drop_duplicates(key + ["measurement_value"])
    stats["exact_duplicates_removed"] = n_records - len(records)
    conflicts = int(records.duplicated(key).sum())
    if conflicts:
        raise ValueError(
            f"tag {tag_id!r}: {conflicts} records with the same time and sensor"
            " but different values"
        )

    dst = records.pivot(
        index="timestamp_utc", columns="measurement_type", values="measurement_value"
    )
    missing = {"temperature", "pressure"} - set(dst.columns)
    if missing:
        raise ValueError(f"tag {tag_id!r} has no {sorted(missing)} records")

    pressure_units = sorted(
        records.loc[records["measurement_type"] == "pressure", "measurement_unit"]
        .astype(str)
        .unique()
    )
    if len(pressure_units) > 1:
        raise ValueError(
            f"tag {tag_id!r}: pressure records with several units {pressure_units}"
        )
    (unit,) = pressure_units
    stats["pressure_unit"] = _PRESSURE_UNITS.get(unit, f"{unit} (as labelled in ETN)")
    if unit not in _PRESSURE_UNITS:
        warnings.warn(
            f"tag {tag_id!r}: unknown pressure unit {unit!r} in ETN,"
            " the values are not converted",
            UserWarning,
        )
    elif unit == "Pa" and dst["pressure"].max() > _MAX_DBAR:
        stats["pressure_unit"] = "Pa"
        warnings.warn(
            f"tag {tag_id!r}: pressure labelled 'Pa' with values up to"
            f" {dst['pressure'].max()}: these look like pascals, not dbar,"
            " and are not depths in metres",
            UserWarning,
        )

    dst = dst[["temperature", "pressure"]].sort_index()
    dst.columns.name = None
    dst.index = pd.DatetimeIndex(dst.index, name="time")
    if dst.index.tz is not None:
        dst.index = dst.index.tz_convert(None)
    return dst, stats


def _naive_utc(time):
    time = pd.Timestamp(time)
    return time.tz_convert(None) if time.tzinfo is not None else time


def _position(animal, prefixes):
    """First complete (longitude, latitude) among ``<prefix>_longitude/latitude``."""
    for prefix in prefixes:
        lon = animal.get(f"{prefix}_longitude")
        lat = animal.get(f"{prefix}_latitude")
        if (
            lon is not None
            and lat is not None
            and not pd.isna(lon)
            and not pd.isna(lat)
        ):
            return (float(lon), float(lat)), f"ETN {prefix}_longitude/latitude"
    return (np.nan, np.nan), None


def open_etn_archival(
    project,
    tag_id,
    *,
    root=ETN_PARQUET_ROOT,
    storage_options=None,
    release_position=None,
    end_time=None,
    end_position=None,
    animals=None,
):
    """Open an ETN archival tag as a tag tree, like :func:`pangeo_fish.io.open_tag`.

    The release time and position come from the ETN animals table. When the
    release position is incomplete (e.g. latitude missing), the surgery and then
    the capture position are used, with a warning. The end of the track
    (``fish_death`` event) is ``end_time``, else the ETN recapture time (the
    retrieval of the tag, which may be after the death of the fish), else the
    last record. The ETN export has no recapture position: ``end_position``
    defaults to ``NA``. The source of every value is stored in the
    ``event_sources`` attribute, and the unit of ``pressure`` (derived from the
    ETN label) in the ``pressure_unit`` attribute.

    A tag attached to several animals of ``project`` raises an error: pass
    ``animals`` restricted to the right animal.

    Use :func:`pangeo_fish.io.write_tag` to write the tag to the tag folder format.

    Parameters
    ----------
    project : str
        ETN project code (e.g. ``"shad_scheldt_dst"``).
    tag_id : str
        Tag serial number (e.g. ``"A19845"``).
    root : str, default: ETN_PARQUET_ROOT
        Root of the ETN parquet export (or of a local copy with the same layout).
    storage_options : mapping, optional
        Passed to :py:func:`fsspec.open`.
    release_position, end_position : tuple of float, optional
        ``(longitude, latitude)`` overriding the release / end positions.
    end_time : str or datetime-like, optional
        Time of the ``fish_death`` event (UTC if time zone naive).
    animals : pandas.DataFrame, optional
        Pre-loaded ETN animals table (see :func:`read_etn_animals`).

    Returns
    -------
    tag : xarray.DataTree
        Tree with the metadata as attributes of the root node and the nodes
        ``dst`` and ``tagging_events``.
    """
    if animals is None:
        animals = read_etn_animals(root, storage_options)
    matches = animals[
        (animals["animal_project_code"] == project)
        & (animals["tag_serial_number"] == tag_id)
    ]
    if matches.empty:
        raise ValueError(
            f"tag {tag_id!r} not found in the ETN animals table of project {project!r}"
        )
    animal_ids = matches["animal_id"].unique()
    if len(animal_ids) > 1:
        # the release time and position depend on the animal: do not guess
        raise ValueError(
            f"tag {tag_id!r} is attached to several animals of project {project!r}"
            f" ({sorted(animal_ids.tolist())}): pass the animals table"
            " restricted to one of them as `animals`"
        )
    animal = matches.iloc[0]

    dst, stats = _read_records(project, tag_id, root, storage_options)

    sources = {}
    release_time = animal["release_date_time"]
    if pd.isna(release_time):
        release_time = dst.index[0]
        sources["release_time"] = "first record (no release time in ETN)"
    else:
        sources["release_time"] = "ETN release_date_time"
    release_time = _naive_utc(release_time)

    if release_position is not None:
        sources["release_position"] = "user"
    else:
        release_position, source = _position(animal, ["release", "surgery", "capture"])
        sources["release_position"] = source or "unknown (NA)"
        if source != "ETN release_longitude/latitude":
            warnings.warn(
                f"tag {tag_id!r}: incomplete release position in ETN,"
                f" using: {sources['release_position']}",
                UserWarning,
            )

    if end_time is not None:
        sources["end_time"] = "user"
    elif not pd.isna(animal.get("recapture_date_time")):
        end_time = animal["recapture_date_time"]
        sources["end_time"] = "ETN recapture_date_time (retrieval of the tag)"
    else:
        end_time = dst.index[-1]
        sources["end_time"] = "last record (no recapture time in ETN)"
    end_time = _naive_utc(end_time)

    if end_position is not None:
        sources["end_position"] = "user"
    else:
        end_position = (np.nan, np.nan)
        sources["end_position"] = "unknown (NA)"

    tagging_events = pd.DataFrame(
        {
            "time": [release_time, end_time],
            "latitude": [release_position[1], end_position[1]],
            "longitude": [release_position[0], end_position[0]],
        },
        index=pd.Index(["release", "fish_death"], name="event_name"),
    ).astype({"latitude": float, "longitude": float})

    try:
        sensor_ranges = _read_sensor_ranges(tag_id, root, storage_options)
    except (OSError, KeyError, ValueError) as e:
        warnings.warn(f"could not read the sensor ranges: {e}", UserWarning)
        sensor_ranges = {}

    def _str_or_none(value):
        return None if value is None or pd.isna(value) else str(value)

    metadata = {
        "pit_tag_id": tag_id,
        "acoustic_tag_id": _str_or_none(animal.get("acoustic_tag_id")),
        "scientific_name": _str_or_none(animal.get("scientific_name")),
        "common_name": _str_or_none(animal.get("common_name")),
        "tagging_campaign": project,
        "etn_animal_id": int(animal["animal_id"]),
        "source": _archival_data_url(project, root),
        "sensor_ranges": sensor_ranges,
        "event_sources": sources,
        "dst_records": len(dst),
        **stats,
    }

    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs=metadata),
            "dst": dst.to_xarray(),
            "tagging_events": tagging_events.to_xarray(),
        }
    )
