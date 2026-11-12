import json

import numpy as np
import pandas as pd
import pytest

from pangeo_fish.etn_archival import (
    open_etn_archival,
    read_etn_animals,
    read_etn_archival_records,
    write_tag,
)
from pangeo_fish.io import open_tag

PROJECT = "dummy_dst"


@pytest.fixture
def etn_root(tmp_path):
    """Small local copy of the ETN parquet export, with the ETN quirks."""
    times = pd.date_range("2023-05-03T08:00", periods=4, freq="h")
    records = []
    for tag, label in [("A1", "Pressure"), ("A2", "pressure")]:
        for i, t in enumerate(times):
            records.append([tag, t, "Temperature", 10.0 + i, "C"])
            records.append([tag, t, label, 1.5 + i, "Pa"])
    records = pd.DataFrame(
        records,
        columns=[
            "tag_id",
            "timestamp_utc",
            "measurement_type",
            "measurement_value",
            "measurement_unit",
        ],
    )
    # exact duplicates of the first tag
    records = pd.concat([records, records[records["tag_id"] == "A1"]])
    records = records.sort_values(["tag_id", "timestamp_utc"])
    records.insert(0, "_id", [f"id{i}" for i in range(len(records))])
    records["csv_source"] = "file.csv"
    path = tmp_path / "archival" / "archival_det" / PROJECT
    path.mkdir(parents=True)
    records.to_parquet(path / "part-0.parquet", index=False, row_group_size=8)

    animal = {
        "animal_id": 1,
        "animal_project_code": PROJECT,
        "tag_serial_number": "A1",
        "tag_type": "archival",
        "acoustic_tag_id": None,
        "scientific_name": "Alosa fallax",
        "common_name": "Twaite shad",
        "capture_latitude": 51.2,
        "capture_longitude": 4.2,
        "surgery_latitude": np.nan,
        "surgery_longitude": np.nan,
        "release_date_time": pd.Timestamp("2023-05-03T09:00", tz="UTC"),
        "release_latitude": 51.1,
        "release_longitude": 4.1,
        "recapture_date_time": pd.Timestamp("2023-05-03T10:30", tz="UTC"),
    }
    second = animal | {
        "animal_id": 2,
        "tag_serial_number": "A2",
        "release_latitude": np.nan,
        "recapture_date_time": pd.NaT,
    }
    # the ETN animals table has duplicated rows
    animals = pd.DataFrame([animal, animal, second])
    path = tmp_path / "acoustic" / "animals"
    path.mkdir(parents=True)
    animals.to_parquet(path / "part-0.parquet", index=False)

    tags = pd.DataFrame(
        {
            "tag_serial_number": ["A1", "A1"],
            "sensor_type": ["temperature", "pressure"],
            "sensor_range_min": ["2", "0"],
            "sensor_range_max": ["34", "300"],
        }
    )
    path = tmp_path / "archival" / "tags"
    path.mkdir(parents=True)
    tags.to_parquet(path / "part-0.parquet", index=False)

    return str(tmp_path)


def test_read_etn_animals(etn_root):
    animals = read_etn_animals(etn_root)
    assert len(animals) == 2


def test_read_records(etn_root):
    dst, stats = read_etn_archival_records(PROJECT, "A1", root=etn_root)

    assert list(dst.columns) == ["temperature", "pressure"]
    assert dst.index.name == "time"
    assert dst.index.tz is None
    assert len(dst) == 4
    assert stats["etn_records"] == 16
    assert stats["exact_duplicates_removed"] == 8
    assert stats["etn_units"] == ["Pressure: Pa", "Temperature: C"]
    # pressure values are kept as is (dbar)
    np.testing.assert_allclose(dst["pressure"], [1.5, 2.5, 3.5, 4.5])

    # other spelling of the sensor name
    dst, _ = read_etn_archival_records(PROJECT, "A2", root=etn_root)
    assert dst["pressure"].notna().all()


def test_read_records_errors(etn_root):
    with pytest.raises(ValueError, match="no sensor records"):
        read_etn_archival_records(PROJECT, "A3", root=etn_root)


def test_open_etn_archival(etn_root):
    tag = open_etn_archival(PROJECT, "A1", root=etn_root)

    assert set(tag.children) == {"dst", "tagging_events"}
    events = tag["tagging_events"].to_dataset()
    assert list(events["event_name"].values) == ["release", "fish_death"]
    release = events.sel(event_name="release")
    assert pd.Timestamp(release["time"].values) == pd.Timestamp("2023-05-03T09:00")
    assert release["latitude"].item() == 51.1
    end = events.sel(event_name="fish_death")
    assert pd.Timestamp(end["time"].values) == pd.Timestamp("2023-05-03T10:30")
    assert np.isnan(end["latitude"].item())

    assert tag.attrs["etn_animal_id"] == 1
    assert tag.attrs["sensor_ranges"] == {
        "temperature": [2.0, 34.0],
        "pressure": [0.0, 300.0],
    }
    assert "recapture" in tag.attrs["event_sources"]["end_time"]


def test_open_etn_archival_missing_release(etn_root):
    with pytest.warns(UserWarning, match="incomplete release position"):
        tag = open_etn_archival(PROJECT, "A2", root=etn_root)
    events = tag["tagging_events"].to_dataset()
    # fallback to the capture position
    assert events["latitude"].sel(event_name="release").item() == 51.2
    # no recapture: last record
    end_time = pd.Timestamp(events["time"].sel(event_name="fish_death").values)
    assert end_time == pd.Timestamp("2023-05-03T11:00")

    tag = open_etn_archival(
        PROJECT,
        "A2",
        root=etn_root,
        release_position=(4.0, 51.0),
        end_time="2023-05-03T10:00",
    )
    events = tag["tagging_events"].to_dataset()
    assert events["latitude"].sel(event_name="release").item() == 51.0
    assert tag.attrs["event_sources"]["end_time"] == "user"


def test_open_etn_archival_unknown_tag(etn_root):
    with pytest.raises(ValueError, match="not found"):
        open_etn_archival("other_project", "A1", root=etn_root)


def test_write_tag_roundtrip(etn_root, tmp_path):
    tag = open_etn_archival(PROJECT, "A1", root=etn_root)
    folder = write_tag(tag, tmp_path / "tags", "A1")

    with open(f"{folder}/tagging_events.csv") as f:
        lines = f.read().splitlines()
    assert lines[0] == "event_name,time,latitude,longitude"
    assert lines[2] == "fish_death,2023-05-03T10:30:00Z,NA,NA"
    with open(f"{folder}/metadata.json") as f:
        assert json.load(f)["pit_tag_id"] == "A1"

    reopened = open_tag(str(tmp_path / "tags"), "A1")
    assert reopened["dst"].to_dataset().equals(tag["dst"].to_dataset())
    np.testing.assert_array_equal(
        reopened["tagging_events/time"].values, tag["tagging_events/time"].values
    )
    assert reopened.attrs["sensor_ranges"] == tag.attrs["sensor_ranges"]
