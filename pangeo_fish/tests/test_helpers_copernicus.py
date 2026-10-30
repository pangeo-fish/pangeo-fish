from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from pangeo_fish import helpers
from pangeo_fish.io import prepare_dataset

IBI = "cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m"
SSH = "cmems_mod_ibi_phy-ssh_my_0.027deg_P1D-m"
STATIC = "cmems_mod_ibi_phy_my_0.027deg-3D_static"
GLOBAL = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
IBI_HOURLY = "cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m"
IBI_HOURLY_SSH = "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m"
IBI_HOURLY_STATIC = "cmems_mod_ibi_phy_anfc_0.027deg-3D_static"
NWS_HOURLY = "cmems_mod_nws_phy-tem_anfc_1.5km-3D_PT1H-i"
NWS_HOURLY_SSH = "cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT1H-i"
NWS_HOURLY_STATIC = "cmems_mod_nws_phy_anfc_1.5km_static"
BBOX = {"latitude": [48, 49], "longitude": [-5, -4], "max_depth": 20}
PERIOD = slice("2021-06-01", "2021-06-03")


@pytest.fixture
def fields():
    coords = {
        "time": np.array(
            ["2021-06-01", "2021-06-02", "2021-06-03"], dtype="datetime64[ns]"
        ),
        "depth": [0.49402538, 10.0, 30.0],
        "latitude": [48.025, 48.05],
        "longitude": [-4.999, -4.971],
    }
    ds = xr.Dataset(
        {
            "thetao": (
                ("time", "depth", "latitude", "longitude"),
                np.full((3, 3, 2, 2), 14.0),
            ),
            "zos": (("time", "latitude", "longitude"), np.full((3, 2, 2), 0.2)),
            "deptho": (("latitude", "longitude"), [[100.0, np.nan], [100.0, 100.0]]),
            "mask": (
                ("depth", "latitude", "longitude"),
                np.tile([[1, 0], [1, 1]], (3, 1, 1)),
            ),
        },
        coords=coords,
    )

    ds["thetao"].attrs["units"] = "degree_Celsius"
    return ds


@pytest.fixture
def client(monkeypatch, fields):
    datasets = {
        IBI: fields[["thetao"]],
        SSH: fields[["zos"]],
        STATIC: fields[["deptho", "mask"]],
        GLOBAL: fields[["thetao", "zos"]],
        GLOBAL.rsplit("_", 1)[0] + "_static": fields[["deptho", "mask"]],
    }
    mock = Mock(side_effect=lambda **kwargs: datasets[kwargs["dataset_id"]])
    monkeypatch.setattr(helpers.copernicusmarine, "open_dataset", mock)
    return mock, datasets


@pytest.mark.parametrize(
    "dataset, expected",
    [
        (IBI, (IBI, SSH, STATIC, "bathy")),
        (GLOBAL, (GLOBAL, GLOBAL, "cmems_mod_glo_phy_my_0.083deg_static", None)),
        (IBI_HOURLY, (IBI_HOURLY, IBI_HOURLY_SSH, IBI_HOURLY_STATIC, "bathy")),
        (NWS_HOURLY, (NWS_HOURLY, NWS_HOURLY_SSH, NWS_HOURLY_STATIC, "bathy")),
        # Do not invent companions for other temperature products.
        (
            "other-temp_P1D-m",
            ("other-temp_P1D-m", "other-temp_P1D-m", "other-temp_static", None),
        ),
    ],
)
def test_resolve_datasets(dataset, expected):
    assert helpers._resolve_copernicus_datasets(dataset) == expected


def open_model(dataset=IBI, **kwargs):
    return helpers._open_copernicus_model(dataset, BBOX, PERIOD, **kwargs)


def test_ibi_requests_and_preparation(client, fields):
    mock, _ = client
    actual = open_model(Username="test-user", Password="test-password")
    expected = prepare_dataset(fields.rename(latitude="lat", longitude="lon"))
    xr.testing.assert_identical(actual, expected)
    calls = [c.kwargs for c in mock.call_args_list]
    assert [(c["dataset_id"], c["variables"]) for c in calls] == [
        (IBI, ["thetao"]),
        (SSH, ["zos"]),
        (STATIC, ["deptho", "mask"]),
    ]
    assert calls[2]["dataset_part"] == "bathy"
    assert "dataset_part" not in calls[0] and "dataset_part" not in calls[1]
    for call in calls:
        assert call["minimum_latitude"] == 48
        assert call["maximum_latitude"] == 49
        assert call["minimum_longitude"] == -5
        assert call["maximum_longitude"] == -4
        assert call["username"] == "test-user"
        assert call["password"] == "test-password"
    for call in calls[:2]:
        assert call["start_datetime"] == "2021-06-01"
        assert call["end_datetime"] == "2021-06-03"
    assert "start_datetime" not in calls[2]
    assert "end_datetime" not in calls[2]
    assert actual["mask"].dims == ("depth", "lat", "lon")


def test_legacy_combined_product(client, fields):
    mock, _ = client
    actual = open_model(GLOBAL)
    xr.testing.assert_identical(
        actual, prepare_dataset(fields.rename(latitude="lat", longitude="lon"))
    )
    calls = [c.kwargs for c in mock.call_args_list]
    assert len(calls) == 2
    assert calls[0]["variables"] == ["thetao", "zos"]
    assert calls[1]["dataset_id"] == "cmems_mod_glo_phy_my_0.083deg_static"
    assert "dataset_part" not in calls[1]


@pytest.mark.parametrize(
    "dataset,variable",
    [(IBI, "thetao"), (SSH, "zos"), (STATIC, "deptho"), (STATIC, "mask")],
)
def test_missing_variable(client, dataset, variable):
    _, datasets = client
    datasets[dataset] = datasets[dataset].drop_vars(variable)
    with pytest.raises(
        ValueError, match=f"{dataset!r}.*missing required variables.*{variable}"
    ):
        open_model()


@pytest.mark.parametrize(
    "dataset,coordinate", [(SSH, "latitude"), (STATIC, "longitude"), (STATIC, "depth")]
)
def test_roundoff_only(client, fields, dataset, coordinate):
    _, datasets = client
    ds = datasets[dataset]
    datasets[dataset] = ds.assign_coords(
        {coordinate: ds[coordinate].astype("float32").astype("float64")}
    )
    xr.testing.assert_identical(
        open_model(), prepare_dataset(fields.rename(latitude="lat", longitude="lon"))
    )


@pytest.mark.parametrize(
    "dataset,coordinate", [(SSH, "latitude"), (STATIC, "longitude"), (STATIC, "depth")]
)
def test_shifted_grid_rejected(client, dataset, coordinate):
    _, datasets = client
    ds = datasets[dataset]
    datasets[dataset] = ds.assign_coords({coordinate: ds[coordinate] + 0.01})
    with pytest.raises(ValueError, match=f"{coordinate} grid differs"):
        open_model()


@pytest.mark.parametrize("change", ["offset", "missing", "duplicate", "reverse"])
def test_incompatible_times_rejected(client, change):
    _, datasets = client
    ds = datasets[SSH]
    if change == "offset":
        ds = ds.assign_coords(time=ds.time + np.timedelta64(1, "h"))
    elif change == "missing":
        ds = ds.isel(time=slice(1, None))
    elif change == "duplicate":
        ds = ds.isel(time=[0, 0, 2])
    else:
        ds = ds.isel(time=slice(None, None, -1))
    datasets[SSH] = ds
    with pytest.raises(ValueError, match="time.*grid"):
        open_model()


@pytest.mark.parametrize("coordinate", ["latitude", "longitude", "depth"])
def test_missing_grid_point_rejected(client, coordinate):
    _, datasets = client
    datasets[STATIC] = datasets[STATIC].isel({coordinate: slice(1, None)})
    with pytest.raises(ValueError, match=f"incompatible {coordinate} grid"):
        open_model()


def test_tag_time_fallback(client, fields):
    mock, _ = client
    helpers._open_copernicus_model(IBI, BBOX, None, tag_log=fields)
    assert mock.call_args_list[0].kwargs["start_datetime"] == "2021-06-01"
    assert mock.call_args_list[0].kwargs["end_datetime"] == "2021-06-03"
    with pytest.raises(ValueError, match="time_slice.*tag_log"):
        helpers._open_copernicus_model(IBI, BBOX, None)


@pytest.mark.parametrize("dataset", [IBI, GLOBAL])
def test_load_model_and_compute_diff(client, fields, dataset):
    tag = xr.Dataset(
        {
            "temperature": ("time", [13.0, 13.0, 13.0]),
            "pressure": ("time", [5.0, 5.0, 5.0]),
        },
        coords={"time": fields.time},
    )
    model = helpers.load_model(
        uri=None,
        tag_log=tag,
        time_slice=PERIOD,
        bbox=BBOX,
        chunk_time=1,
        remote_options={},
        copernicus_model_name=dataset,
    )
    assert set(model.data_vars) == {
        "TEMP",
        "XE",
        "H0",
        "mask",
        "dynamic_depth",
        "dynamic_bathymetry",
    }
    assert model.sizes == {"time": 3, "depth": 2, "lat": 2, "lon": 2}
    assert model.chunks["time"] == (1, 1, 1)
    assert model.latitude.dims == model.longitude.dims == ("lat", "lon")
    xr.testing.assert_allclose(
        model.dynamic_depth,
        (model.depth + model.XE).transpose(*model.dynamic_depth.dims),
    )
    diff, _ = helpers.compute_diff(
        reference_model=model, tag_log=tag, relative_depth_threshold=0, chunk_time=1
    )
    assert diff["diff"].dims == ("time", "lat", "lon")
    assert diff["diff"].notnull().any().compute().item()


@pytest.mark.parametrize(
    "temperature, ssh, static",
    [
        (IBI_HOURLY, IBI_HOURLY_SSH, IBI_HOURLY_STATIC),
        (NWS_HOURLY, NWS_HOURLY_SSH, NWS_HOURLY_STATIC),
    ],
)
def test_hourly_requests_use_hourly_bounds(client, fields, temperature, ssh, static):
    mock, datasets = client
    hours = np.arange("2021-06-01T00", "2021-06-04T00", dtype="datetime64[h]").astype(
        "datetime64[ns]"
    )
    hourly = fields.isel(time=[0] * hours.size).assign_coords(time=hours)
    datasets[temperature] = hourly[["thetao"]]
    datasets[ssh] = hourly[["zos"]]
    datasets[static] = hourly[["deptho", "mask"]]

    actual = helpers._open_copernicus_model(
        temperature, BBOX, slice("2021-06-01", "2021-06-03")
    )

    xr.testing.assert_identical(
        actual, prepare_dataset(hourly.rename(latitude="lat", longitude="lon"))
    )
    calls = [c.kwargs for c in mock.call_args_list]
    assert [c["dataset_id"] for c in calls] == [temperature, ssh, static]
    assert calls[2]["dataset_part"] == "bathy"
    assert calls[0]["start_datetime"] == "2021-06-01T00:00:00"
    # a date-only stop includes the whole last day
    assert calls[0]["end_datetime"] == "2021-06-03T23:59:59"
    # SSH is requested over the time range of the temperature
    assert calls[1]["start_datetime"] == "2021-06-01T00:00:00"
    assert calls[1]["end_datetime"] == "2021-06-03T23:00:00"


@pytest.fixture
def clipping_client(monkeypatch, fields):
    """SSH covers 5 days, temperature only the last 3 (as for NWS/IBI hourly)."""
    days = np.arange("2021-05-30", "2021-06-04", dtype="datetime64[D]").astype(
        "datetime64[ns]"
    )
    full = fields.isel(time=[0] * days.size).assign_coords(time=days)
    datasets = {
        IBI: full[["thetao"]].sel(time=slice("2021-06-01", None)),
        SSH: full[["zos"]],
        STATIC: full[["deptho", "mask"]],
    }

    def open_dataset(**kwargs):
        ds = datasets[kwargs["dataset_id"]]
        if "start_datetime" in kwargs:
            ds = ds.sel(time=slice(kwargs["start_datetime"], kwargs["end_datetime"]))
        return ds

    mock = Mock(side_effect=open_dataset)
    monkeypatch.setattr(helpers.copernicusmarine, "open_dataset", mock)
    return mock


def test_ssh_clipped_to_temperature_period(clipping_client):
    with pytest.warns(UserWarning, match="only covers 2021-06-01.*2021-06-03"):
        actual = helpers._open_copernicus_model(
            IBI, BBOX, slice("2021-05-30", "2021-06-03")
        )
    assert actual.time.size == 3
    assert str(actual.time.values[0])[:10] == "2021-06-01"
    ssh_call = clipping_client.call_args_list[1].kwargs
    assert (ssh_call["start_datetime"], ssh_call["end_datetime"]) == (
        "2021-06-01",
        "2021-06-03",
    )


def test_no_warning_when_period_covered(clipping_client):
    actual = helpers._open_copernicus_model(
        IBI, BBOX, slice("2021-06-01", "2021-06-03")
    )
    assert actual.time.size == 3


@pytest.mark.parametrize("land_value", [np.nan, 0])
def test_land_depth_is_masked(client, fields, land_value):
    _, datasets = client
    static = datasets[STATIC].copy(deep=True)
    static["deptho"] = static["deptho"].fillna(21.2)
    static["mask"] = static["mask"].where(static["mask"] > 0, land_value)
    datasets[STATIC] = static
    actual = open_model()
    assert np.isnan(actual["H0"].isel(lat=0, lon=1)).all()
    assert (actual["H0"].isel(lat=0, lon=0) == 100.0).all()


def test_land_depth_masked_for_legacy_product(client, fields):
    _, datasets = client
    key = GLOBAL.rsplit("_", 1)[0] + "_static"
    datasets[key] = datasets[key].assign(deptho=datasets[key]["deptho"].fillna(5.0))
    actual = open_model(GLOBAL)
    assert np.isnan(actual["H0"].isel(lat=0, lon=1)).all()


def test_hourly_time_bounds_keep_explicit_timestamps():
    start, end = helpers._copernicus_time_bounds(
        NWS_HOURLY, slice("2025-01-01T06:00", "2025-01-02T18:00"), None
    )
    assert (start, end) == ("2025-01-01T06:00:00", "2025-01-02T18:00:00")


def test_hourly_time_bounds_from_tag_log():
    tag_log = xr.Dataset(
        coords={
            "time": np.array(
                ["2025-01-01T06:30", "2025-01-02T18:15"], dtype="datetime64[ns]"
            )
        }
    )
    start, end = helpers._copernicus_time_bounds(IBI_HOURLY, None, tag_log)
    assert (start, end) == ("2025-01-01T06:30:00", "2025-01-02T18:15:00")


def test_daily_time_bounds_unchanged():
    assert helpers._copernicus_time_bounds(IBI, PERIOD, None) == (
        "2021-06-01",
        "2021-06-03",
    )
    with pytest.raises(ValueError, match="time_slice"):
        helpers._copernicus_time_bounds(IBI, None, None)
