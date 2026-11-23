import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.describe import _records_summary, describe_tag
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
    elif end == "fridge":
        # recovered and stored: in-air pressure reading, constant cold temperature
        pressure[after] = 1.8 + rng.normal(0, 0.02, after.sum())
        temperature[after] = 4 + rng.normal(0, 0.1, after.sum())

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

    def test_tag_kept_in_a_fridge_is_reported(self):
        description = describe_tag(synthetic_tag(end="fridge"), tide=False)
        suggested = description["suggested_end"]

        # the pressure back at the in-air reading does not end the track: it is
        # reported, for the data owner to check against the recovery of the tag
        assert suggested["end"] is None
        reported = suggested["evidence"]["pressure_at_air_level_from"]
        assert abs(reported - END) <= pd.Timedelta(1, "h")
        assert "in-air reading" in suggested["warning"]

    def test_end_uses_the_pre_release_level(self, monkeypatch):
        import pangeo_fish.describe as describe_module

        received = {}
        detect_end = describe_module.detect_end_of_track

        def spy(records, **kwargs):
            received.update(kwargs)
            return detect_end(records, **kwargs)

        monkeypatch.setattr(describe_module, "detect_end_of_track", spy)
        description = describe_tag(synthetic_tag(end="fridge"), tide=False)

        level = description["suggested_start"]["pre_release_level"]
        assert received["air_level"] == level == pytest.approx(1.8, abs=0.1)

    def test_end_kwargs_override_the_air_level(self):
        description = describe_tag(
            synthetic_tag(end="fridge"), tide=False, end_kwargs={"air_level": 10.0}
        )

        suggested = description["suggested_end"]
        assert suggested["parameters"]["air_level"] == 10.0
        assert suggested["evidence"]["pressure_at_air_level_from"] is None
        assert suggested["warning"] is None

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

    def test_tide_on_the_seabed_after_the_death(self):
        # a dead fish on the seabed does not end the track: no end, and the tide
        # recorded after the death is part of the track
        description = describe_tag(synthetic_tag("seabed"))
        tide = description["tide"]

        assert description["suggested_end"]["end"] is None
        assert tide["available"]
        assert "fraction_after_end" not in tide
        assert tide["fraction_last_3_days"] > 0.5

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


def test_records_summary_drops_missing_times_and_sorts():
    times = pd.DatetimeIndex(
        [
            "2023-05-03 08:05",
            "2023-05-03 08:00",
            pd.NaT,
            "2023-05-03 08:01",
            "2023-05-03 08:02",
            "2023-05-03 08:03",
        ],
        name="time",
    )
    df = pd.DataFrame({"pressure": [5.0, 0.0, 9.0, 1.0, 2.0, 3.0]}, index=times)

    summary = _records_summary(df, gap_factor=1.5)

    assert summary["n_missing_times"] == 1
    assert summary["n_records"] == 5
    assert summary["first_record"] == pd.Timestamp("2023-05-03 08:00")
    assert summary["last_record"] == pd.Timestamp("2023-05-03 08:05")
    assert summary["duration_days"] == pytest.approx(5 / 1440)
    assert summary["sampling_interval"] == pd.Timedelta(1, "min")
    assert summary["n_gaps"] == 1
    assert summary["longest_gap"] == pd.Timedelta(2, "min")


@pytest.mark.parametrize("name", ["start_kwargs", "end_kwargs"])
@pytest.mark.parametrize("key", ["sensor_ranges", "sensor_range_tolerance"])
def test_rejects_flag_arguments_in_bounds_kwargs(name, key):
    with pytest.raises(ValueError, match=f"{name} cannot contain .*{key}"):
        describe_tag(synthetic_tag(), tide=False, **{name: {key: None}})
