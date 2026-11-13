import numpy as np
import pandas as pd
import pytest

from pangeo_fish.etn_acoustic import find_etn_acoustic_detections


def _write(df, root, path):
    path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def etn_root(tmp_path):
    """Small local copy of the ETN parquet export."""
    animals = pd.DataFrame(
        {
            "animal_id": [1, 1, 2, 3, 4, 5, 6],
            "animal_project_code": [
                "eels",
                "eels",
                "eels",
                "eels",
                "closed",
                "eels",
                "eels",
            ],
            "tag_serial_number": ["111", "111", "A2", "333", "444", "555", "555"],
            "tag_type": [
                "acoustic-archival",
                "acoustic-archival",
                "archival",
                "acoustic-archival",
                "acoustic-archival",
                "acoustic",
                "acoustic",
            ],
            "acoustic_tag_id": [
                "A69-1",
                "A69-1",
                None,
                "A69-3",
                "A69-4",
                "A69-5",
                "A69-5",
            ],
            "acoustic_tag_id_alternative": [
                "A69-0",
                "A69-0",
                None,
                None,
                None,
                None,
                None,
            ],
        }
    )
    # duplicated rows, as in the ETN animals table
    _write(animals, tmp_path, "acoustic/animals/part-0.parquet")

    detections = pd.DataFrame(
        {
            "acoustic_project_code": "receivers",
            "acoustic_tag_id": ["A69-1", "A69-1", "A69-1", "A69-1", "A69-9"],
            "animal_id": [1, 1, 1, 1, 9],
            "animal_project_code": "eels",
            "date_time": pd.to_datetime(
                [
                    "2020-01-02T10:00",
                    "2020-01-01T10:00",
                    "2020-01-03T10:00",
                    "2020-01-04T10:00",
                    "2020-01-01T00:00",
                ]
            ),
            "deploy_latitude": [51.0, 51.0, 52.0, 53.0, 50.0],
            "deploy_longitude": [3.0, 3.0, 4.0, 5.0, 2.0],
            # deployment 40 is not in the deployments table
            "deployment_id": [10, 10, 20, 40, 30],
            "detection_id": [1, 2, 3, 4, 5],
            "receiver_id": ["VR2W-1", "VR2W-1", "VR2W-2", "VR2W-4", "VR2W-3"],
            "station_name": ["s10", "s10", "s20", "s40", "s30"],
            "tag_serial_number": ["111", "111", "111", "111", "999"],
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
            # deployment 20 has no recover time
            "recover_date_time": pd.to_datetime(
                ["2020-02-01", None, "2020-02-01"], utc=True
            ),
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
    with pytest.warns(UserWarning) as record:
        result = find_etn_acoustic_detections("eels", "111", root=str(etn_root))
    messages = [str(w.message) for w in record]
    assert any("deployments [40] are not in the ETN deployments" in m for m in messages)
    assert any(
        "first or last detection" in m and "recover_time: [20, 40]" in m
        for m in messages
    )

    assert set(result.children) == {"acoustic", "stations"}
    assert result.attrs == {
        "etn_animal_id": 1,
        "animal_project_code": "eels",
        "double_tagged": True,
        "acoustic_tag_ids": ["A69-0", "A69-1"],
    }

    acoustic = result["acoustic"].to_dataset()
    assert set(acoustic.data_vars) == {"deployment_id", "longitude", "latitude"}
    assert acoustic.sizes["time"] == 4
    assert acoustic.indexes["time"].is_monotonic_increasing
    assert acoustic.indexes["time"].tz is None

    stations = result["stations"].to_dataset()
    assert list(stations["deployment_id"].values) == [10, 20, 40]
    assert stations["deploy_latitude"].sel(deployment_id=20).item() == 52.0
    assert stations["deploy_time"].dtype.kind == "M"
    assert stations["recover_time"].dtype.kind == "M"

    def time(variable, deployment_id):
        return pd.Timestamp(stations[variable].sel(deployment_id=deployment_id).item())

    # ETN times are kept
    assert time("deploy_time", 10) == pd.Timestamp("2019-12-01")
    assert time("recover_time", 10) == pd.Timestamp("2020-02-01")
    # missing times: first / last detection at the deployment
    assert time("deploy_time", 20) == pd.Timestamp("2019-12-01")
    assert time("recover_time", 20) == pd.Timestamp("2020-01-03T10:00")
    assert time("deploy_time", 40) == pd.Timestamp("2020-01-04T10:00")
    assert time("recover_time", 40) == pd.Timestamp("2020-01-04T10:00")
    assert stations["deploy_latitude"].sel(deployment_id=40).item() == 53.0


@pytest.mark.parametrize(
    ["project", "tag_id", "double_tagged", "match"],
    [
        ("eels", "A2", False, "no acoustic tag"),
        ("eels", "333", True, "0 detections"),
        ("closed", "444", True, "moratorium"),
    ],
)
def test_nothing_found(etn_root, project, tag_id, double_tagged, match):
    with pytest.warns(UserWarning, match=match):
        result = find_etn_acoustic_detections(project, tag_id, root=str(etn_root))

    assert not result.children
    assert result.attrs["double_tagged"] is double_tagged
    if not double_tagged:
        assert result.attrs["acoustic_tag_ids"] == []


@pytest.mark.parametrize(
    ["project", "tag_id", "match"],
    [
        ("eels", "42", "not found"),
        # the tag exists, but in another project
        ("closed", "111", "not found"),
        ("eels", "555", r"several animals .*\[5, 6\]"),
    ],
)
def test_select_animal_errors(etn_root, project, tag_id, match):
    with pytest.raises(ValueError, match=match):
        find_etn_acoustic_detections(project, tag_id, root=str(etn_root))
