import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.describe import describe_tag, format_description

RELEASE = pd.Timestamp("2023-05-03 11:07")
END = pd.Timestamp("2023-05-08 08:17")
SENSOR_RANGES = {"pressure": [0, 300], "temperature": [2, 34]}


def synthetic_records(end="stranded", seed=0):
    """1-min records: 3 h in air (1.8 m), swimming fish, then stranded / seabed."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2023-05-03 08:00", END + pd.Timedelta(days=3), freq="1min")
    hours = ((time - time[0]) / pd.Timedelta("1h")).to_numpy()

    pressure = 10 + 4 * np.sin(2 * np.pi * hours / 0.7) + rng.normal(0, 1.5, len(time))
    temperature = 15 + 0.02 * hours + rng.normal(0, 0.1, len(time))

    before = time < RELEASE
    pressure[before] = 1.8 + rng.normal(0, 0.03, before.sum())
    temperature[before] = 12 + rng.normal(0, 0.3, before.sum())

    after = time >= END
    hours_after = ((time[after] - END) / pd.Timedelta("1h")).to_numpy()
    if end == "stranded":
        pressure[after] = 5.4 + rng.normal(0, 0.02, after.sum())
        temperature[after] = 19.5 + 4.5 * np.sin(2 * np.pi * (hours_after - 6) / 24)
        # picked up: hot, in the hand
        picked_up = time >= END + pd.Timedelta(days=2)
        temperature[picked_up] = 36
    elif end == "seabed":
        pressure[after] = 25 + 2.5 * np.sin(2 * np.pi * hours_after / 12.42)
        pressure[after] += rng.normal(0, 0.02, after.sum())
        temperature[after] = 16 + rng.normal(0, 0.05, after.sum())

    pressure = np.round(pressure / 0.0625) * 0.0625
    return xr.Dataset(
        {"pressure": ("time", pressure), "temperature": ("time", temperature)},
        coords={"time": time.to_numpy()},
    )


def synthetic_tag(end="stranded", events=None):
    if events is None:
        events = {
            "release": (RELEASE, 51.07, 4.19),
            "fish_death": (END + pd.Timedelta("1D 10h"), np.nan, np.nan),
        }
    tagging_events = xr.Dataset(
        {
            "time": ("event_name", [value[0] for value in events.values()]),
            "latitude": ("event_name", [value[1] for value in events.values()]),
            "longitude": ("event_name", [value[2] for value in events.values()]),
        },
        coords={"event_name": list(events)},
    )
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"pit_tag_id": "T1", "sensor_ranges": SENSOR_RANGES}),
            "dst": synthetic_records(end=end),
            "tagging_events": tagging_events,
        }
    )


@pytest.fixture(scope="module")
def stranded():
    return describe_tag(synthetic_tag("stranded"))


def test_describe_tag_records(stranded):
    records = stranded["records"]

    assert stranded["tag_name"] == "T1"
    assert records["first_record"] == pd.Timestamp("2023-05-03 08:00")
    assert records["sampling_interval"] == pd.Timedelta("1min")
    assert records["regular_sampling"]
    assert records["n_gaps"] == 0
    assert records["n_duplicate_times"] == 0
    assert records["n_missing"] == {"pressure": 0, "temperature": 0}


def test_describe_tag_ranges(stranded):
    track = stranded["ranges"]["track"]
    record = stranded["ranges"]["record"]

    assert track["temperature"]["max"] < 20
    assert record["temperature"]["max"] == pytest.approx(36)
    out = stranded["out_of_sensor_range"]["temperature"]
    assert out["n_records"] > 0
    assert out["first_after_start"] == END + pd.Timedelta(days=2)


def test_describe_tag_suggestions(stranded):
    assert abs(stranded["suggested_start"]["start"] - RELEASE) <= pd.Timedelta("5min")
    assert abs(stranded["suggested_end"]["end"] - END) <= pd.Timedelta("5min")
    comparison = stranded["comparison"]
    assert abs(comparison["release_minus_suggested_start_days"]) < 0.01
    assert comparison["fish_death_minus_suggested_end_days"] == pytest.approx(
        1.42, abs=0.01
    )


def test_describe_tag_pressure_offset(stranded):
    offset = stranded["pressure_offset"]

    assert offset["offset"] == pytest.approx(1.8, abs=0.1)
    assert offset["warnings"] == []


def test_describe_tag_questions(stranded):
    questions = "\n".join(stranded["questions"])

    assert "which end should be used" in questions
    assert "correct the pressure" in questions
    assert "Where was the fish released" not in questions
    assert "seabed" not in questions


def test_describe_tag_seabed_tide():
    description = describe_tag(synthetic_tag("seabed"))

    assert description["tide"]["available"]
    assert description["tide"]["fraction_after_end"] > 0.5
    assert any("seabed" in question for question in description["questions"])


def test_describe_tag_without_events():
    description = describe_tag(
        synthetic_tag(events={"release": (RELEASE, np.nan, np.nan)}), tide=False
    )

    questions = "\n".join(description["questions"])
    assert "Where was the fish released" in questions
    assert "is this the end of the track" in questions
    assert description["tide"]["available"] is False


def test_describe_tag_dataset():
    description = describe_tag(
        synthetic_records(), tag_name="T2", sensor_ranges=SENSOR_RANGES, tide=False
    )

    assert description["tag_name"] == "T2"
    assert description["tagging_events"]["times"] == {}
    assert description["suggested_end"]["end"] is not None


def test_format_description(stranded):
    text = format_description(stranded)

    assert text.startswith("## Tag T1")
    assert "suggested end" in text
    assert "### Questions" in text
