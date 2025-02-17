import types

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from pangeo_fish.tracks import additional_quantities, to_dataframe, to_trajectory


class DummyDataset:
    """
    Simulate a dataset that has a to_pandas() method.
    """

    def __init__(self, data):
        self.data = data

    def to_pandas(self):
        df = pd.DataFrame(self.data)
        # Add a DatetimeIndex to meet the requirements of movingpandas.Trajectory.
        df.index = pd.date_range("2022-01-01", periods=len(df), freq="H")
        return df


class Trajectory:
    """
    Fake class to simulate mpd.Trajectory.
    """

    def __init__(self, df, traj_id, x, y, crs="epsg:4326", **kwargs):
        if not isinstance(df, gpd.GeoDataFrame):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError("The DataFrame must have a DatetimeIndex.")
            df = gpd.GeoDataFrame(
                df.drop(columns=[x, y]),
                geometry=[Point(xy) for xy in zip(df[x], df[y])],
                crs=crs,
                index=df.index,
            )
        if len(df) < 2:
            raise ValueError("The DataFrame must have at least two rows.")
        self.df = df.copy()
        self.traj_id = traj_id
        self.x = x
        self.y = y

    def copy(self):
        import copy

        return copy.deepcopy(self)

    def add_speed(self, name, units):
        # For testing, add a "speed" column with the fixed value 42.
        self.df[name] = 42
        return self

    def add_distance(self, name, units):
        # For testing, add a "distance" column with the fixed value 24.
        self.df[name] = 24
        return self


# Create a fake module "mpd" and assign our Trajectory class to it.
mpd = types.ModuleType("mpd")
mpd.Trajectory = Trajectory


class DummyGeoSeries:
    """
    Simulate an object similar to a GeoSeries that has a get_coordinates() method.
    """

    def __init__(self, coords_df):
        self.coords_df = coords_df

    def get_coordinates(self):
        return self.coords_df


class DummyGeoDataFrame(pd.DataFrame):
    """
    Simulate a GeoDataFrame.
    """

    _metadata = ["_geometry"]

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        # Store the geometry object in the _geometry attribute.
        self._geometry = value
        # If the value has a get_coordinates method, create a "geometry" column.
        if hasattr(value, "get_coordinates"):
            coords = value.get_coordinates()  # DataFrame with 'x' and 'y'
            self["geometry"] = [Point(x, y) for x, y in zip(coords["x"], coords["y"])]
        else:
            self["geometry"] = value

    def merge(self, right, left_index=True, right_index=True):
        # Use the standard pandas merge method.
        return pd.merge(self, right, left_index=left_index, right_index=right_index)


@pytest.fixture
def dummy_dataset():
    """
    Create a dummy dataset with longitude and latitude data.
    """
    data = {"longitude": [0, 1, 2], "latitude": [10, 11, 12]}
    return DummyDataset(data)


@pytest.fixture
def dummy_geo_series():
    """
    Create a dummy GeoSeries with coordinate data.
    """
    coords_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    return DummyGeoSeries(coords_df)


@pytest.fixture
def dummy_geodataframe(dummy_geo_series):
    """
    Create a dummy GeoDataFrame with extra data.

    The dummy GeoDataFrame contains a 'traj_id' column and other columns.
    The geometry is assigned using the provided dummy GeoSeries.
    """
    df = pd.DataFrame({"traj_id": ["a", "b", "c"], "other": [10, 20, 30]})
    gdf = DummyGeoDataFrame(df)
    gdf.geometry = dummy_geo_series
    return gdf


def test_to_trajectory(dummy_dataset):
    """
    Test conversion of a dummy dataset to a trajectory.
    """
    ds = dummy_dataset
    traj = to_trajectory(ds, name="test_traj")

    expected_df = ds.to_pandas()
    expected_geo = gpd.GeoDataFrame(
        expected_df.drop(columns=["longitude", "latitude"]),
        geometry=[
            Point(xy) for xy in zip(expected_df["longitude"], expected_df["latitude"])
        ],
        crs="epsg:4326",
        index=expected_df.index,
    )
    print("expected", expected_geo)
    print("traj.df", traj.df)

    pd.testing.assert_series_equal(
        traj.df["geometry"].reset_index(drop=True),
        expected_geo["geometry"].reset_index(drop=True),
    )


def test_additional_quantities_empty():
    """
    Test additional_quantities with an empty list.
    """
    df = pd.DataFrame({"longitude": [0, 1, 2], "latitude": [10, 11, 12]})
    df.index = pd.date_range("2022-01-01", periods=len(df), freq="H")
    traj = mpd.Trajectory(df, traj_id="dummy", x="longitude", y="latitude")
    new_traj = additional_quantities(traj, [])
    pd.testing.assert_frame_equal(new_traj.df, traj.df)


def test_additional_quantities_speed_distance():
    """
    Test additional_quantities for speed and distance.
    """
    df = pd.DataFrame({"longitude": [0, 1, 2], "latitude": [10, 11, 12]})
    df.index = pd.date_range("2022-01-01", periods=len(df), freq="H")
    traj = mpd.Trajectory(df, traj_id="dummy", x="longitude", y="latitude")
    new_traj = additional_quantities(traj, ["speed", "distance"])

    assert "speed" in new_traj.df.columns
    assert "distance" in new_traj.df.columns
    assert all(new_traj.df["speed"] == 42)
    assert all(new_traj.df["distance"] == 24)


def test_additional_quantities_unknown():
    """
    Test additional_quantities with an unknown quantity.
    """
    df = pd.DataFrame({"longitude": [0, 1, 2], "latitude": [10, 11, 12]})
    df.index = pd.date_range("2022-01-01", periods=len(df), freq="H")
    traj = mpd.Trajectory(df, traj_id="dummy", x="longitude", y="latitude")
    with pytest.raises(ValueError, match="unknown quantity: unknown"):
        additional_quantities(traj, ["unknown"])


def test_to_dataframe(dummy_geodataframe):
    """
    Test conversion of a GeoDataFrame to a regular DataFrame.
    """
    gdf = dummy_geodataframe
    result = to_dataframe(gdf)

    expected = pd.DataFrame(
        {"other": [10, 20, 30], "longitude": [1, 2, 3], "latitude": [4, 5, 6]}
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
