import json

import fsspec
import pytest

from pangeo_fish.io import open_tag


@pytest.fixture
def dummy_mapper():
    """
    Fixture to create a dummy mapper with in-memory test data
    """

    files = {
        "tag_dummy/dst.csv": b"""time,temperature,pressure
    2022-06-13T00:00:00+00:00,20.0,1.5
    2022-06-13T01:00:00+00:00,21.0,1.6
    2022-06-13T02:00:00+00:00,22.0,1.7
    2022-06-13T03:00:00+00:00,23.0,1.8
    2022-06-13T04:00:00+00:00,23.0,1.9
    """,
        "tag_dummy/tagging_events.csv": b"""event_name,time,longitude,latitude
    release,2022-06-13T11:40:30+00:00,-5.098,48.45
    fish_death,2022-06-13T12:00:00+00:00,-5.098,48.45
    """,
        "stations.csv": b"""deployment_id,station_name,deploy_time,recover_time,deploy_longitude,deploy_latitude
    28689,station_1,2022-06-12T12:00:00+00:00,2022-06-20T12:00:00+00:00,-5.1,48.45
    """,
        "tag_dummy/acoustic.csv": b"""time,deployment_id
    2022-06-13T00:00:00+00:00,28689
    """,
        "tag_dummy/metadata.json": json.dumps({"dummy_key": "dummy_value"}).encode(
            "utf-8"
        ),
    }

    fs = fsspec.filesystem("memory")
    mapper = fs.get_mapper()
    mapper.update(files)
    return mapper


def test_open_tag(dummy_mapper):
    """
    Test opening a tag with open_tag
    """

    tag = open_tag(dummy_mapper, "tag_dummy")
    assert set(tag.keys()).issuperset({"dst", "tagging_events", "stations", "acoustic"})
    assert tag["/"].attrs.get("dummy_key") == "dummy_value"
    assert tag["dst/temperature"].sel(time="2022-06-13T00:00:00").item() == 20.0
    assert tag["dst/temperature"].sel(time="2022-06-13T01:00:00").item() == 21.0
