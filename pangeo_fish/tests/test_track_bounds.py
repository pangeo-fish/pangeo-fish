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


def tag_with_phases(*, rest_days, final):
    """Active fish, ``rest_days`` days at constant depth in the water, then ``final``.

    ``final`` is "air" (diurnal temperature cycle), "sensor" (pressure out of the
    sensor range, plausible temperature) or "water" (3 more days in the water).
    The fish dies at STRANDED; the tag leaves the water ``rest_days`` later.
    """
    rng = np.random.default_rng(0)
    active = synthetic_tag(end="none")
    active = active.sel(time=active["time"] < np.datetime64(STRANDED))
    rest_end = STRANDED + pd.Timedelta(rest_days, "D")
    time = pd.date_range(STRANDED, rest_end + pd.Timedelta(3, "D"), freq="1min")
    n = len(time)
    pressure = np.full(n, 25.0) + rng.normal(0, 0.02, n)
    temperature = 16 + rng.normal(0, 0.05, n)
    after = time >= rest_end
    hours = ((time[after] - rest_end) / pd.Timedelta(1, "h")).to_numpy()
    if final == "air":
        pressure[after] = 5.4 + rng.normal(0, 0.02, after.sum())
        temperature[after] = 19.5 + 4.5 * np.sin(2 * np.pi * (hours - 6) / 24)
    elif final == "sensor":
        pressure[after] = 245.81
    tail = xr.Dataset(
        {
            "pressure": ("time", quantize(pressure)),
            "temperature": ("time", temperature),
        },
        coords={"time": time.to_numpy()},
    )
    return xr.concat([active, tail], dim="time")


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
        assert actual["inactive_period"] is None
        assert "stays in the water until the last record" in actual["reason"]

    def test_active_then_out_of_water(self):
        ds = tag_with_phases(rest_days=0, final="air")
        actual = track_bounds.detect_end_of_track(ds)

        assert abs(actual["end"] - STRANDED) <= pd.Timedelta(5, "min")
        assert actual["likely_cause"] == "out_of_water"
        assert actual["inactive_period"] is None

    def test_death_then_out_of_water_ends_at_the_exit(self):
        ds = tag_with_phases(rest_days=5, final="air")
        actual = track_bounds.detect_end_of_track(ds, start=RELEASE)

        exit_time = STRANDED + pd.Timedelta(5, "D")
        assert abs(actual["end"] - exit_time) <= pd.Timedelta(1, "h")
        assert actual["likely_cause"] == "out_of_water"
        # the death does not end the track, it starts the inactive period
        first, last = actual["inactive_period"]
        assert abs(first - STRANDED) <= pd.Timedelta(1, "h")
        assert last <= actual["end"]
        assert last - first >= pd.Timedelta(4, "D")

    def test_death_in_water_until_the_last_record(self):
        ds = tag_with_phases(rest_days=5, final="water")
        actual = track_bounds.detect_end_of_track(ds, start=RELEASE)

        assert actual["end"] is None
        assert actual["likely_cause"] is None
        first, last = actual["inactive_period"]
        assert abs(first - STRANDED) <= pd.Timedelta(1, "h")
        assert last == ds["time"].values[-1]
        assert "stays in the water" in actual["reason"]

    def test_seabed_tag_is_not_an_end(self):
        ds = synthetic_tag(end="seabed")
        actual = track_bounds.detect_end_of_track(ds, start=RELEASE)

        assert actual["end"] is None
        assert abs(actual["inactive_period"][0] - STRANDED) <= pd.Timedelta(1, "h")

    def test_out_of_sensor_range(self):
        ds = tag_with_phases(rest_days=0, final="sensor")
        # the temperature stays plausible: only the pressure is out of range
        ds.attrs["sensor_ranges"] = {"pressure": [0, 200], "temperature": [2, 34]}
        actual = track_bounds.detect_end_of_track(ds)

        assert abs(actual["end"] - STRANDED) <= pd.Timedelta(5, "min")
        assert actual["likely_cause"] == "out_of_sensor_range"

    def test_rest_then_activity_is_not_an_end(self):
        ds = synthetic_tag(end="none")
        rest = slice(STRANDED, STRANDED + pd.Timedelta(2, "D"))
        ds["pressure"].loc[{"time": rest}] = 25.0
        actual = track_bounds.detect_end_of_track(ds, start=RELEASE)

        assert actual["end"] is None
        first, last = actual["inactive_period"]
        assert last - first >= pd.Timedelta(1, "D")

    def test_short_rest_is_ignored(self):
        ds = synthetic_tag(end="none")
        # 3 h without vertical movement in the middle of the track
        rest = slice("2023-05-05 02:00", "2023-05-05 05:00")
        ds["pressure"].loc[{"time": rest}] = 12.0
        actual = track_bounds.detect_end_of_track(ds)

        assert actual["end"] is None

    def test_resumed_activity_warning(self):
        ds = synthetic_tag(end="stranded")
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


def _track_bounds_warnings(function, ds, **kwargs):
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        function(ds, **kwargs)
    return [str(w.message) for w in record if "track_bounds:" in str(w.message)]


@pytest.mark.parametrize("step", [1, 10])
@pytest.mark.parametrize("end", ["stranded", "seabed", "none"])
@pytest.mark.parametrize(
    "function",
    [track_bounds.detect_start_of_track, track_bounds.detect_end_of_track],
)
def test_no_warning_for_clean_records(function, end, step):
    ds = synthetic_tag(end=end).isel(time=slice(None, None, step))

    assert _track_bounds_warnings(function, ds) == []


@pytest.mark.parametrize(
    "function",
    [track_bounds.detect_start_of_track, track_bounds.detect_end_of_track],
)
def test_warns_when_sampling_is_too_coarse_for_the_window(function):
    ds = synthetic_tag().isel(time=slice(None, None, 30))

    with pytest.warns(UserWarning, match=r"sampling interval \(0 days 00:30:00\).*1h"):
        function(ds)
    # a longer window is fine
    assert _track_bounds_warnings(function, ds, window="3h") == []


def _noisy_rest_tag(noise):
    ds = synthetic_tag(end="seabed")
    rng = np.random.default_rng(1)
    rest = ds["time"].to_numpy() >= np.datetime64(STRANDED)
    ds["pressure"].values[rest] += rng.normal(0, noise, rest.sum())
    return ds


@pytest.mark.parametrize(
    "function",
    [track_bounds.detect_start_of_track, track_bounds.detect_end_of_track],
)
def test_warns_when_noise_is_above_the_activity_threshold(function):
    ds = _noisy_rest_tag(0.3)

    with pytest.warns(UserWarning, match=r"activity_threshold = 0\.1 m"):
        function(ds)
    # raising the threshold above the noise silences it
    assert _track_bounds_warnings(function, ds, activity_threshold=1.0) == []


def test_end_without_start_warns_once():
    ds = _noisy_rest_tag(0.3)

    messages = _track_bounds_warnings(track_bounds.detect_end_of_track, ds)

    assert len(messages) == 1
