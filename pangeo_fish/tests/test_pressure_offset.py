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
    hours = ((time - time[0]) / pd.Timedelta("1h")).to_numpy()
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


def test_estimate_constant_offset():
    ds = synthetic_tag()
    actual = estimate_pressure_offset(
        ds,
        in_air=[
            (None, RELEASE - pd.Timedelta("1min")),
            (RECOVERY + pd.Timedelta("1min"), None),
        ],
    )

    assert actual["offset"] == pytest.approx(1.8, abs=0.02)
    assert len(actual["segments"]) == 2
    assert all(segment["used"] for segment in actual["segments"])
    assert actual["spread"] < 0.05
    assert actual["fraction_negative_after_correction"] == 0
    assert actual["warnings"] == []


@pytest.mark.parametrize(
    "in_air",
    [
        ("2023-05-03 08:00", "2023-05-03 11:00"),
        slice("2023-05-03 08:00", "2023-05-03 11:00"),
        [slice(None, "2023-05-03 11:00")],
    ],
)
def test_estimate_offset_in_air_forms(in_air):
    ds = synthetic_tag()
    actual = estimate_pressure_offset(ds, in_air=in_air)

    assert actual["offset"] == pytest.approx(1.8, abs=0.02)
    assert len(actual["segments"]) == 1


def test_estimate_offset_mask():
    ds = synthetic_tag()
    mask = ds["time"] < np.datetime64(RELEASE)
    actual = estimate_pressure_offset(ds, in_air=mask)

    assert actual["offset"] == pytest.approx(1.8, abs=0.02)
    assert actual["segments"][0]["stop"] == RELEASE - pd.Timedelta("1min")


def test_estimate_offset_rejects_in_water_period():
    ds = synthetic_tag()
    actual = estimate_pressure_offset(
        ds, in_air=[(None, "2023-05-03 11:00"), ("2023-05-04", "2023-05-04 06:00")]
    )

    assert [segment["used"] for segment in actual["segments"]] == [True, False]
    assert actual["offset"] == pytest.approx(1.8, abs=0.02)
    assert any("not used" in warning for warning in actual["warnings"])


def test_estimate_offset_no_valid_period():
    ds = synthetic_tag()
    actual = estimate_pressure_offset(ds, in_air=("2023-05-04", "2023-05-04 06:00"))

    assert actual["offset"] is None
    with pytest.raises(ValueError, match="no offset"):
        correct_pressure_offset(ds, actual)


def test_estimate_offset_drift():
    ds = synthetic_tag(offset_before=1.8, offset_after=2.6)
    in_air = [
        (None, RELEASE - pd.Timedelta("1min")),
        (RECOVERY + pd.Timedelta("1min"), None),
    ]

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


def test_estimate_offset_invalid_combine():
    with pytest.raises(ValueError, match="combine"):
        estimate_pressure_offset(synthetic_tag(), in_air=(None, RELEASE), combine="max")


def test_correct_pressure_offset():
    ds = synthetic_tag()
    corrected = correct_pressure_offset(ds, 1.8)

    np.testing.assert_allclose(corrected["pressure"], ds["pressure"] - 1.8)
    assert corrected["pressure"].attrs["pressure_offset"] == 1.8
    assert corrected["pressure"].attrs["units"] == "m"
    assert "pressure_offset" not in ds["pressure"].attrs

    with pytest.raises(ValueError, match="already corrected"):
        correct_pressure_offset(corrected, 1.8)


def test_estimate_offset_datatree():
    ds = synthetic_tag()
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "dst": ds})
    actual = estimate_pressure_offset(tree, in_air=(None, "2023-05-03 11:00"))

    assert actual["offset"] == pytest.approx(1.8, abs=0.02)


def test_estimate_offset_rejects_sensor_failure():
    ds = synthetic_tag()
    ds["pressure"].loc[{"time": slice(RECOVERY + pd.Timedelta("1min"), None)}] = 245.81
    actual = estimate_pressure_offset(
        ds, in_air=[(None, "2023-05-03 11:00"), (RECOVERY + pd.Timedelta("1min"), None)]
    )

    assert [segment["used"] for segment in actual["segments"]] == [True, False]
    assert actual["offset"] == pytest.approx(1.8, abs=0.02)


def test_estimate_offset_in_water_check_period():
    ds = synthetic_tag()
    # after the recovery the tag is stored and reads 0.5 m (drift)
    after = slice(RECOVERY + pd.Timedelta("1min"), None)
    ds["pressure"].loc[{"time": after}] = 0.5
    in_air = (None, "2023-05-03 11:00")

    everything = estimate_pressure_offset(ds, in_air=in_air)
    track = estimate_pressure_offset(ds, in_air=in_air, in_water=(RELEASE, RECOVERY))

    assert everything["fraction_negative_after_correction"] > 0.2
    assert track["fraction_negative_after_correction"] == 0
    assert track["warnings"] == []
