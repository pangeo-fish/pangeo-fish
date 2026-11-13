import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.describe import describe_tag
from pangeo_fish.pressure_offset import estimate_pressure_offset

RELEASE = pd.Timestamp("2023-05-03 11:07")
END = pd.Timestamp("2023-05-08 08:17")
SENSOR_RANGES = {"pressure": [0, 300], "temperature": [2, 34]}


def synthetic_records(end="stranded", seed=0):
    """1-min records: 3 h in air (1.8 m), swimming fish, then stranded / seabed."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2023-05-03 08:00", END + pd.Timedelta(3, "D"), freq="1min")
    hours = ((time - time[0]) / pd.Timedelta(1, "h")).to_numpy()

    pressure = 10 + 4 * np.sin(2 * np.pi * hours / 0.7) + rng.normal(0, 1.5, len(time))
    temperature = 15 + 0.02 * hours + rng.normal(0, 0.1, len(time))

    before = time < RELEASE
    pressure[before] = 1.8 + rng.normal(0, 0.03, before.sum())
    temperature[before] = 12 + rng.normal(0, 0.3, before.sum())

    after = time >= END
    hours_after = ((time[after] - END) / pd.Timedelta(1, "h")).to_numpy()
    if end == "stranded":
        pressure[after] = 5.4 + rng.normal(0, 0.02, after.sum())
        temperature[after] = 19.5 + 4.5 * np.sin(2 * np.pi * (hours_after - 6) / 24)
        # picked up: hot, in the hand
        picked_up = time >= END + pd.Timedelta(2, "D")
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


def synthetic_tag(end="stranded", events=None, sensor_ranges=SENSOR_RANGES):
    if events is None:
        events = {
            "release": (RELEASE, 51.07, 4.19),
            "fish_death": (END + pd.Timedelta(34, "h"), np.nan, np.nan),
        }
    tagging_events = xr.Dataset(
        {
            "time": ("event_name", [value[0] for value in events.values()]),
            "latitude": ("event_name", [value[1] for value in events.values()]),
            "longitude": ("event_name", [value[2] for value in events.values()]),
        },
        coords={"event_name": list(events)},
    )
    attrs = {"pit_tag_id": "T1"}
    if sensor_ranges is not None:
        attrs["sensor_ranges"] = sensor_ranges
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs=attrs),
            "dst": synthetic_records(end=end),
            "tagging_events": tagging_events,
        }
    )


class TestDescribeTag:
    def test_records(self):
        description = describe_tag(synthetic_tag(), tide=False)
        records = description["records"]

        assert description["tag_name"] == "T1"
        assert records["first_record"] == pd.Timestamp("2023-05-03 08:00")
        assert records["sampling_interval"] == pd.Timedelta(1, "min")
        assert records["regular_sampling"]
        assert records["n_gaps"] == 0
        assert records["n_duplicate_times"] == 0
        assert records["n_missing"] == {"pressure": 0, "temperature": 0}

    def test_ranges_on_the_suggested_track(self):
        tag = synthetic_tag()
        description = describe_tag(tag, tide=False)
        start = description["suggested_start"]["start"]
        end = description["suggested_end"]["end"]
        temperature = tag["dst"]["temperature"]

        record = description["ranges"]["record"]["temperature"]
        track = description["ranges"]["track"]["temperature"]
        assert record["max"] == float(temperature.max())
        assert track["max"] == float(temperature.sel(time=slice(start, end)).max())
        assert track["min"] == float(temperature.sel(time=slice(start, end)).min())

    @pytest.mark.parametrize(
        ["sensor_ranges", "expected_first"],
        [
            pytest.param(SENSOR_RANGES, END + pd.Timedelta(2, "D"), id="with-ranges"),
            pytest.param(None, None, id="without-ranges"),
        ],
    )
    def test_out_of_sensor_range(self, sensor_ranges, expected_first):
        description = describe_tag(
            synthetic_tag(sensor_ranges=sensor_ranges), tide=False
        )
        actual = description["out_of_sensor_range"]

        if expected_first is None:
            assert actual is None
        else:
            assert actual["sensor_ranges"] == SENSOR_RANGES
            assert actual["n_records"] > 0
            assert actual["first_after_start"] == expected_first

    def test_comparison_is_event_minus_suggestion(self):
        fish_death = END + pd.Timedelta(34, "h")
        description = describe_tag(synthetic_tag(), tide=False)
        start = description["suggested_start"]["start"]
        end = description["suggested_end"]["end"]
        comparison = description["comparison"]

        assert comparison["release_minus_suggested_start_days"] == pytest.approx(
            (RELEASE - start) / pd.Timedelta(1, "D")
        )
        assert comparison["fish_death_minus_suggested_end_days"] == pytest.approx(
            (fish_death - end) / pd.Timedelta(1, "D")
        )

    def test_pressure_offset_uses_the_pre_release_period(self):
        tag = synthetic_tag()
        description = describe_tag(tag, tide=False)
        start = description["suggested_start"]
        end = description["suggested_end"]["end"]

        expected = estimate_pressure_offset(
            tag,
            in_air=slice(*start["pre_release_period"]),
            in_water=slice(start["start"], end),
        )
        actual = description["pressure_offset"]
        assert actual["offset"] == expected["offset"]
        assert actual["segments"] == expected["segments"]
        assert (
            actual["fraction_negative_after_correction"]
            == expected["fraction_negative_after_correction"]
        )

    def test_tide_after_the_end(self):
        description = describe_tag(synthetic_tag("seabed"))
        tide = description["tide"]

        assert tide["available"]
        assert tide["fraction_after_end"] > 0.5

    def test_tide_not_computed(self):
        description = describe_tag(synthetic_tag(), tide=False)

        assert description["tide"] == {"available": False, "reason": "not computed"}

    def test_tagging_events(self):
        events = {"release": (RELEASE, np.nan, np.nan)}
        description = describe_tag(synthetic_tag(events=events), tide=False)

        assert description["tagging_events"] == {
            "times": {"release": RELEASE},
            "positions": {"release": None},
        }
        assert list(description["comparison"]) == ["release_minus_suggested_start_days"]

    def test_dataset(self):
        description = describe_tag(
            synthetic_records(), tag_name="T2", sensor_ranges=SENSOR_RANGES, tide=False
        )

        assert description["tag_name"] == "T2"
        assert description["tagging_events"] == {"times": {}, "positions": {}}
        assert description["comparison"] == {}
        assert "questions" not in description
