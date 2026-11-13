import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.pressure_offset import (
    correct_pressure_offset,
    estimate_pressure_offset,
)

RELEASE = pd.Timestamp("2023-05-03 11:07")
RECOVERY = pd.Timestamp("2023-05-06 12:00")


def synthetic_tag(offset_before=1.8, offset_after=1.8, seed=0):
    """1-min records: in air, at sea (5-15 m), in air again."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2023-05-03 08:00", "2023-05-07 12:00", freq="1min")
    hours = ((time - time[0]) / pd.Timedelta(1, "h")).to_numpy()
    depth = 10 + 5 * np.sin(2 * np.pi * hours / 3)
    drift = np.interp(
        time.asi8, [time.asi8[0], time.asi8[-1]], [offset_before, offset_after]
    )
    in_air = (time < RELEASE) | (time > RECOVERY)
    depth[in_air] = 0
    pressure = depth + drift + rng.normal(0, 0.03, len(time))
    return xr.Dataset(
        {"pressure": ("time", pressure, {"units": "m"})},
        coords={"time": time.to_numpy()},
    )


BEFORE_RELEASE = slice(None, RELEASE - pd.Timedelta(1, "min"))
AFTER_RECOVERY = slice(RECOVERY + pd.Timedelta(1, "min"), None)


class TestEstimatePressureOffset:
    def test_constant(self):
        ds = synthetic_tag()
        actual = estimate_pressure_offset(ds, in_air=[BEFORE_RELEASE, AFTER_RECOVERY])

        assert actual["offset"] == pytest.approx(1.8, abs=0.02)
        assert len(actual["segments"]) == 2
        assert all(segment["used"] for segment in actual["segments"])
        assert actual["spread"] < 0.05
        assert actual["fraction_negative_after_correction"] == 0
        assert actual["warnings"] == []

    @pytest.mark.parametrize(
        "in_air",
        [
            pytest.param(slice("2023-05-03 08:00", "2023-05-03 11:00"), id="slice"),
            pytest.param([slice(None, "2023-05-03 11:00")], id="list-of-slices"),
        ],
    )
    def test_in_air_forms(self, in_air):
        ds = synthetic_tag()
        actual = estimate_pressure_offset(ds, in_air=in_air)

        assert actual["offset"] == pytest.approx(1.8, abs=0.02)
        assert len(actual["segments"]) == 1

    @pytest.mark.parametrize(
        "in_air",
        [
            pytest.param(
                ["2023-05-03T08:00", "2023-05-03T11:06"], id="list-of-two-strings"
            ),
            pytest.param(("2023-05-03T08:00", "2023-05-03T11:06"), id="tuple"),
            pytest.param([(None, "2023-05-03T11:06")], id="list-of-tuples"),
        ],
    )
    def test_rejects_other_in_air_forms(self, in_air):
        with pytest.raises(TypeError, match="slice"):
            estimate_pressure_offset(synthetic_tag(), in_air=in_air)

    def test_mask(self):
        ds = synthetic_tag()
        mask = ds["time"] < np.datetime64(RELEASE)
        actual = estimate_pressure_offset(ds, in_air=mask)

        assert actual["offset"] == pytest.approx(1.8, abs=0.02)
        assert actual["segments"][0]["stop"] == RELEASE - pd.Timedelta(1, "min")

    def test_rejects_non_boolean_mask(self):
        ds = synthetic_tag()
        with pytest.raises(TypeError, match="boolean"):
            estimate_pressure_offset(ds, in_air=ds["pressure"])

    @pytest.mark.parametrize(
        ["modify", "rejected"],
        [
            pytest.param(None, slice("2023-05-04", "2023-05-04 06:00"), id="in-water"),
            pytest.param(AFTER_RECOVERY, AFTER_RECOVERY, id="sensor-failure"),
        ],
    )
    def test_rejects_period(self, modify, rejected):
        ds = synthetic_tag()
        if modify is not None:
            ds["pressure"].loc[{"time": modify}] = 245.81
        actual = estimate_pressure_offset(ds, in_air=[BEFORE_RELEASE, rejected])

        assert [segment["used"] for segment in actual["segments"]] == [True, False]
        assert actual["offset"] == pytest.approx(1.8, abs=0.02)
        assert any("not used" in warning for warning in actual["warnings"])

    def test_no_valid_period(self):
        ds = synthetic_tag()
        actual = estimate_pressure_offset(
            ds, in_air=slice("2023-05-04", "2023-05-04 06:00")
        )

        assert actual["offset"] is None
        with pytest.raises(ValueError, match="no offset"):
            correct_pressure_offset(ds, actual)

    def test_drift(self):
        ds = synthetic_tag(offset_before=1.8, offset_after=2.6)
        in_air = [BEFORE_RELEASE, AFTER_RECOVERY]

        median = estimate_pressure_offset(ds, in_air=in_air)
        assert median["spread"] == pytest.approx(0.75, abs=0.1)
        assert any("disagree" in warning for warning in median["warnings"])

        first = estimate_pressure_offset(ds, in_air=in_air, combine="first")
        last = estimate_pressure_offset(ds, in_air=in_air, combine="last")
        assert first["offset"] < last["offset"]

        linear = estimate_pressure_offset(ds, in_air=in_air, combine="linear")
        assert isinstance(linear["offset"], xr.DataArray)
        assert linear["offset"].sizes == ds.sizes
        assert float(linear["offset"][0]) == pytest.approx(first["offset"])
        assert float(linear["offset"][-1]) == pytest.approx(last["offset"])

        corrected = correct_pressure_offset(ds, linear)
        in_water = corrected["pressure"].sel(time=slice("2023-05-04", "2023-05-05"))
        assert float(in_water.min()) == pytest.approx(5, abs=0.3)

    @pytest.mark.parametrize("combine", ["first", "last"])
    def test_periods_sorted_by_time(self, combine):
        ds = synthetic_tag(offset_before=1.8, offset_after=2.6)

        in_order = estimate_pressure_offset(
            ds, in_air=[BEFORE_RELEASE, AFTER_RECOVERY], combine=combine
        )
        reversed_ = estimate_pressure_offset(
            ds, in_air=[AFTER_RECOVERY, BEFORE_RELEASE], combine=combine
        )

        assert reversed_["offset"] == in_order["offset"]
        assert reversed_["segments"][0]["stop"] == RELEASE - pd.Timedelta(1, "min")

    def test_invalid_combine(self):
        with pytest.raises(ValueError, match="combine"):
            estimate_pressure_offset(
                synthetic_tag(), in_air=BEFORE_RELEASE, combine="max"
            )

    def test_datatree(self):
        ds = synthetic_tag()
        tree = xr.DataTree.from_dict({"/": xr.Dataset(), "dst": ds})
        actual = estimate_pressure_offset(tree, in_air=BEFORE_RELEASE)

        assert actual["offset"] == pytest.approx(1.8, abs=0.02)

    def test_in_water_check_period(self):
        ds = synthetic_tag()
        # after the recovery the tag is stored and reads 0.5 m (drift)
        ds["pressure"].loc[{"time": AFTER_RECOVERY}] = 0.5
        in_air = slice(None, "2023-05-03 11:00")

        everything = estimate_pressure_offset(ds, in_air=in_air)
        track = estimate_pressure_offset(
            ds, in_air=in_air, in_water=slice(RELEASE, RECOVERY)
        )

        assert everything["fraction_negative_after_correction"] > 0.2
        assert track["fraction_negative_after_correction"] == 0
        assert track["warnings"] == []

    def test_rejects_several_in_water_periods(self):
        ds = synthetic_tag()
        with pytest.raises(ValueError, match="single period"):
            estimate_pressure_offset(
                ds,
                in_air=BEFORE_RELEASE,
                in_water=[slice(RELEASE, "2023-05-04"), slice("2023-05-05", RECOVERY)],
            )


class TestCorrectPressureOffset:
    def test_dataset(self):
        ds = synthetic_tag()
        corrected = correct_pressure_offset(ds, 1.8)

        np.testing.assert_allclose(corrected["pressure"], ds["pressure"] - 1.8)
        assert corrected["pressure"].attrs["pressure_offset"] == 1.8
        assert corrected["pressure"].attrs["units"] == "m"
        assert "pressure_offset" not in ds["pressure"].attrs

        with pytest.raises(ValueError, match="already corrected"):
            correct_pressure_offset(corrected, 1.8)

    def test_datatree(self):
        ds = synthetic_tag()
        tree = xr.DataTree.from_dict(
            {"/": xr.Dataset(attrs={"pit_tag": "A0"}), "dst": ds, "other": xr.Dataset()}
        )
        estimate = estimate_pressure_offset(tree, in_air=BEFORE_RELEASE)
        corrected = correct_pressure_offset(tree, estimate)

        assert isinstance(corrected, xr.DataTree)
        assert set(corrected.children) == {"dst", "other"}
        assert corrected.attrs == {"pit_tag": "A0"}
        np.testing.assert_allclose(
            corrected["dst/pressure"], ds["pressure"] - estimate["offset"]
        )
        assert "pressure_offset" not in tree["dst/pressure"].attrs
