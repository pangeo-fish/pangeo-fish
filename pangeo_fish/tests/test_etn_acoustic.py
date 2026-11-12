import numpy as np
import pandas as pd
import pytest

from pangeo_fish.etn_acoustic import find_etn_acoustic_detections
from pangeo_fish.io import open_tag


def _write(df, root, path):
    path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def etn_root(tmp_path):
    """Small local copy of the ETN parquet export."""
    animals = pd.DataFrame(
        {
            "animal_id": [1, 1, 2, 3, 4],
            "animal_project_code": ["eels", "eels", "eels", "eels", "closed"],
            "tag_serial_number": ["111", "111", "A2", "333", "444"],
            "tag_type": [
                "acoustic-archival",
                "acoustic-archival",
                "archival",
                "acoustic-archival",
                "acoustic-archival",
            ],
            "acoustic_tag_id": ["A69-1", "A69-1", None, "A69-3", "A69-4"],
            "acoustic_tag_id_alternative": ["A69-0", "A69-0", None, None, None],
        }
    )
    # duplicated rows, as in the ETN animals table
    _write(animals, tmp_path, "acoustic/animals/part-0.parquet")

    detections = pd.DataFrame(
        {
            "acoustic_project_code": "receivers",
            "acoustic_tag_id": ["A69-1", "A69-1", "A69-1", "A69-9"],
            "animal_id": [1, 1, 1, 9],
            "animal_project_code": "eels",
            "date_time": pd.to_datetime(
                [
                    "2020-01-02T10:00",
                    "2020-01-01T10:00",
                    "2020-01-03T10:00",
                    "2020-01-01T00:00",
                ]
            ),
            "deploy_latitude": [51.0, 51.0, 52.0, 50.0],
            "deploy_longitude": [3.0, 3.0, 4.0, 2.0],
            "deployment_id": [10, 10, 20, 30],
            "detection_id": [1, 2, 3, 4],
            "receiver_id": ["VR2W-1", "VR2W-1", "VR2W-2", "VR2W-3"],
            "station_name": ["s10", "s10", "s20", "s30"],
            "tag_serial_number": ["111", "111", "111", "999"],
        }
    )
    _write(detections, tmp_path, "acoustic/detections/eels/part-0.parquet")

    deployments = pd.DataFrame(
        {
            "deployment_id": [10, 20, 30],
            "receiver_id": ["VR2W-1", "VR2W-2", "VR2W-3"],
            "acoustic_project_code": "receivers",
            "station_name": ["s10", "s20", "s30"],
            "deploy_date_time": pd.to_datetime(["2019-12-01"] * 3, utc=True),
            "deploy_latitude": [51.0, 52.0, 50.0],
            "deploy_longitude": [3.0, 4.0, 2.0],
            "recover_date_time": pd.to_datetime(["2020-02-01"] * 3, utc=True),
            "recover_latitude": np.nan,
            "recover_longitude": np.nan,
        }
    )
    _write(deployments, tmp_path, "acoustic/deployments/part-0.parquet")

    projects = pd.DataFrame(
        {"project_code": ["eels", "closed"], "moratorium": [False, True]}
    )
    _write(projects, tmp_path, "acoustic/projects/part-0.parquet")

    return tmp_path


def test_find_detections(etn_root):
    result = find_etn_acoustic_detections(animal_id=1, root=str(etn_root))

    assert result.found
    assert result.double_tagged
    assert result.acoustic_tag_ids == ["A69-0", "A69-1"]
    assert result.animal_project_code == "eels"
    assert len(result.detections) == 3
    assert result.detections["time"].is_monotonic_increasing
    assert result.detections["time"].dt.tz is None
    assert list(result.stations.index) == [10, 20]
    assert result.stations.loc[20, "deploy_latitude"] == 52.0
    assert "3 detections" in result.message

    # same animal, by tag serial number
    by_serial = find_etn_acoustic_detections(
        tag_serial_number="111", root=str(etn_root)
    )
    assert by_serial.animal_id == 1


def test_no_acoustic_tag(etn_root):
    result = find_etn_acoustic_detections(tag_serial_number="A2", root=str(etn_root))

    assert not result.found
    assert not result.double_tagged
    assert result.acoustic_tag_ids == []
    assert "no acoustic tag" in result.message


def test_no_detections(etn_root):
    result = find_etn_acoustic_detections(animal_id=3, root=str(etn_root))

    assert not result.found
    assert result.double_tagged
    assert "0 detections" in result.message


def test_missing_detection_file(etn_root):
    result = find_etn_acoustic_detections(animal_id=4, root=str(etn_root))

    assert not result.found
    assert "moratorium" in result.message


def test_unknown_animal(etn_root):
    with pytest.raises(ValueError, match="not found"):
        find_etn_acoustic_detections(animal_id=42, root=str(etn_root))
    with pytest.raises(ValueError, match="exactly one"):
        find_etn_acoustic_detections(root=str(etn_root))


def test_write_and_open_tag(etn_root, tmp_path):
    tag_root = tmp_path / "tags"
    folder = tag_root / "T1"
    folder.mkdir(parents=True)
    (folder / "dst.csv").write_text(
        "time,temperature,pressure\n2020-01-01T00:00:00Z,10.0,1.0\n"
    )
    (folder / "tagging_events.csv").write_text(
        "event_name,time,latitude,longitude\n"
        "release,2020-01-01T00:00:00Z,51.0,3.0\n"
        "fish_death,2020-01-05T00:00:00Z,NA,NA\n"
    )
    (folder / "metadata.json").write_text("{}")
    # an existing stations.csv is merged, not overwritten
    (tag_root / "stations.csv").write_text(
        "deployment_id,station_name,deploy_time,deploy_longitude,deploy_latitude,"
        "recover_time,recover_longitude,recover_latitude\n"
        "5,s5,2019-01-01T00:00:00Z,1.0,49.0,2019-06-01T00:00:00Z,NA,NA\n"
    )

    result = find_etn_acoustic_detections(animal_id=1, root=str(etn_root))
    result.write(tag_root, "T1")

    tag = open_tag(str(tag_root), "T1")
    assert tag["acoustic"].sizes["time"] == 3
    assert set(tag["acoustic/deployment_id"].values) == {10, 20}
    assert list(tag["stations/deployment_id"].values) == [5, 10, 20]
    assert pd.Timestamp(
        tag["stations/deploy_time"].sel(deployment_id=10).values
    ) == pd.Timestamp("2019-12-01")
