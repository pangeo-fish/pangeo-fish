import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish import track_bounds

RELEASE = pd.Timestamp("2023-05-03 11:07")
STRANDED = pd.Timestamp("2023-05-08 08:17")
SENSOR_RANGES = {"pressure": [0, 300], "temperature": [2, 34]}


def quantize(values, step=0.0625):
    return np.round(values / step) * step


def synthetic_tag(*, in_air_before=True, end="stranded", seed=0, sensor_ranges=None):
    """1-min records: in air, swimming fish, then stranded / on the seabed."""
    rng = np.random.default_rng(seed)
    first = pd.Timestamp("2023-05-03 08:00") if in_air_before else RELEASE
    stranded = STRANDED
    time = pd.date_range(first, stranded + pd.Timedelta(3, "D"), freq="1min")
    hours = ((time - time[0]) / pd.Timedelta(1, "h")).to_numpy()

    # swimming fish: depth changing by metres every few minutes
    pressure = 10 + 4 * np.sin(2 * np.pi * hours / 0.7) + rng.normal(0, 1.5, len(time))
    temperature = 15 + 0.02 * hours + rng.normal(0, 0.1, len(time))

    before = time < RELEASE
    pressure[before] = 1.8 + rng.normal(0, 0.03, before.sum())
    temperature[before] = 12 + rng.normal(0, 0.3, before.sum())

    after = time >= stranded
    hours_after = ((time[after] - stranded) / pd.Timedelta(1, "h")).to_numpy()
    if end == "stranded":
        pressure[after] = 5.4 + rng.normal(0, 0.02, after.sum())
        # day / night cycle of a tag in the air
        temperature[after] = 19.5 + 4.5 * np.sin(2 * np.pi * (hours_after - 6) / 24)
    elif end == "seabed":
        # dead fish on the seabed: only the tide
        pressure[after] = 25 + 2.5 * np.sin(2 * np.pi * hours_after / 12.42)
        pressure[after] += rng.normal(0, 0.02, after.sum())
        temperature[after] = 16 + rng.normal(0, 0.05, after.sum())
    elif end == "sensor_failure":
        pressure[after] = 245.81
        temperature[after] = 53.98
    elif end == "none":
        pass

    ds = xr.Dataset(
        {
            "pressure": ("time", quantize(pressure)),
            "temperature": ("time", temperature),
        },
        coords={"time": time.to_numpy()},
    )
    if sensor_ranges is not None:
        ds.attrs["sensor_ranges"] = sensor_ranges
    return ds


def test_flag_records_activity():
    ds = synthetic_tag()
    activity = track_bounds.flag_records(ds)["activity"]

    assert activity.sizes == ds.sizes
    assert float(activity.sel(time=slice("2023-05-04", "2023-05-07")).median()) > 0.5
    assert float(activity.sel(time=slice("2023-05-09", None)).median()) < 0.05


def test_flag_records():
    ds = synthetic_tag(end="sensor_failure", sensor_ranges=SENSOR_RANGES)
    flags = track_bounds.flag_records(ds)

    assert set(flags.data_vars) == {
        "activity",
        "inactive",
        "out_of_sensor_range",
        "temperature_outside_water_range",
        "suspect",
    }
    failure = flags.sel(time=slice(STRANDED + pd.Timedelta(1, "h"), None))
    assert bool(failure["out_of_sensor_range"].all())
    assert bool(failure["temperature_outside_water_range"].all())
    assert not bool(
        flags["out_of_sensor_range"]
        .sel(time=slice(None, STRANDED - pd.Timedelta(1, "min")))
        .any()
    )


def test_detect_start_of_track():
    ds = synthetic_tag()
    actual = track_bounds.detect_start_of_track(ds)

    assert abs(actual["start"] - RELEASE) <= pd.Timedelta(5, "min")
    assert actual["pre_release_period"][0] == pd.Timestamp("2023-05-03 08:00")
    assert actual["pre_release_level"] == pytest.approx(1.8, abs=0.1)


def test_detect_start_of_track_in_water_from_first_record():
    ds = synthetic_tag(in_air_before=False)
    actual = track_bounds.detect_start_of_track(ds)

    assert actual["start"] == RELEASE
    assert actual["pre_release_period"] is None
    assert actual["pre_release_level"] is None


def sensor_failure_tree():
    ds = synthetic_tag(end="sensor_failure")
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"sensor_ranges": SENSOR_RANGES}),
            "dst": ds,
        }
    )


class TestDetectEndOfTrack:
    @pytest.mark.parametrize(
        ["make_tag", "start", "tolerance", "expected_cause"],
        [
            pytest.param(
                lambda: synthetic_tag(end="stranded"),
                None,
                pd.Timedelta(5, "min"),
                "out_of_water",
                id="stranded",
            ),
            pytest.param(
                lambda: synthetic_tag(end="seabed"),
                RELEASE,
                pd.Timedelta(1, "h"),
                "no_vertical_movement",
                id="seabed",
            ),
            pytest.param(
                sensor_failure_tree,
                None,
                pd.Timedelta(5, "min"),
                "out_of_sensor_range",
                id="sensor-failure",
            ),
        ],
    )
    def test_cause(self, make_tag, start, tolerance, expected_cause):
        actual = track_bounds.detect_end_of_track(make_tag(), start=start)

        assert abs(actual["end"] - STRANDED) <= tolerance
        assert actual["likely_cause"] == expected_cause

    def test_stranded_details(self):
        ds = synthetic_tag(end="stranded")
        actual = track_bounds.detect_end_of_track(ds)

        assert actual["end"] < STRANDED
        assert actual["evidence"]["temperature_daily_range_during"] > 5
        assert actual["warning"] is None
        assert actual["parameters"]["min_duration"] == "12h"

    def test_not_found(self):
        ds = synthetic_tag(end="none")
        actual = track_bounds.detect_end_of_track(ds)

        assert actual["end"] is None
        assert actual["likely_cause"] is None
        assert "active until the last record" in actual["reason"]

    def test_short_rest_is_ignored(self):
        ds = synthetic_tag(end="none")
        # 3 h without vertical movement in the middle of the track
        rest = slice("2023-05-05 02:00", "2023-05-05 05:00")
        ds["pressure"].loc[{"time": rest}] = 12.0
        actual = track_bounds.detect_end_of_track(ds)

        assert actual["end"] is None

    def test_resumed_activity_warning(self):
        ds = synthetic_tag(end="seabed")
        # active again for a day after the end
        revival = slice(
            STRANDED + pd.Timedelta(1, "D"), STRANDED + pd.Timedelta(2, "D")
        )
        swimming = synthetic_tag(end="none")["pressure"].sel(time=revival)
        ds["pressure"].loc[{"time": revival}] = swimming.to_numpy()
        actual = track_bounds.detect_end_of_track(ds, start=RELEASE)

        assert abs(actual["end"] - STRANDED) <= pd.Timedelta(1, "h")
        assert actual["warning"] is not None

    def test_start_detection_uses_the_same_thresholds(self, monkeypatch):
        received = {}
        detect_start = track_bounds.detect_start_of_track

        def spy(tag_log, **kwargs):
            received.update(kwargs)
            return detect_start(tag_log, **kwargs)

        monkeypatch.setattr(track_bounds, "detect_start_of_track", spy)
        track_bounds.detect_end_of_track(
            synthetic_tag(), max_gap="10min", level_change_threshold=0.8
        )

        assert received["max_gap"] == "10min"
        assert received["level_change_threshold"] == 0.8
