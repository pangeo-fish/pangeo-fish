import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.tracks import additional_quantities, to_dataframe, to_trajectory


def test_to_trajectory() -> None:
    longitudes = np.array([10, 20, 30], dtype="float64")
    latitudes = np.array([-10, 0, 10], dtype="float64")
    times = pd.date_range("2021-06-22 11:43:35", freq="90 min", periods=3)

    ds = xr.Dataset(
        coords={
            "longitude": ("time", longitudes),
            "latitude": ("time", latitudes),
            "time": times,
        }
    )

    actual = to_trajectory(ds, name="test_traj")

    expected_df = pd.DataFrame(
        {"longitude": longitudes, "latitude": latitudes}, index=times
    )
    expected = mpd.Trajectory(
        expected_df, traj_id="test_traj", x="longitude", y="latitude"
    )

    assert actual == expected


@pytest.mark.parametrize(
    ["quantities", "expected"],
    (
        ([], pd.DataFrame({"x": [0]}).drop(columns=["x"])),
        (
            ["speed", "distance"],
            pd.DataFrame(
                {
                    "speed": [155.620, 155.620, 155.366],
                    "distance": [0.0, 155.620, 155.366],
                }
            ),
        ),
    ),
)
def test_additional_quantities(quantities, expected) -> None:
    df = pd.DataFrame(
        {"longitude": [0, 1, 2], "latitude": [10, 11, 12]},
        index=pd.date_range("2022-01-01", periods=3, freq="h"),
    )
    expected.index = df.index

    traj = mpd.Trajectory(df, traj_id="dummy", x="longitude", y="latitude")
    new_traj = additional_quantities(traj, quantities)

    actual = new_traj.df.drop(columns=["geometry", "traj_id"])
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-4)


def test_additional_quantities_unknown() -> None:
    df = pd.DataFrame(
        {"longitude": [0, 1, 2], "latitude": [10, 11, 12]},
        index=pd.date_range("2022-01-01", periods=3, freq="h"),
    )
    traj = mpd.Trajectory(df, traj_id="dummy", x="longitude", y="latitude")
    with pytest.raises(ValueError, match="unknown quantity: unknown"):
        additional_quantities(traj, ["unknown"])


def test_to_dataframe() -> None:
    x = np.array([1, 2, 3], dtype="float64")
    y = np.array([4, 5, 6], dtype="float64")
    other = np.array([10, 20, 30])

    geometry = gpd.points_from_xy(x=x, y=y, crs="epsg:4326")
    gdf = gpd.GeoDataFrame({"other": other, "traj_id": list("abc")}, geometry=geometry)

    actual = to_dataframe(gdf)

    expected = pd.DataFrame({"other": other, "longitude": x, "latitude": y})
    pd.testing.assert_frame_equal(actual, expected)
