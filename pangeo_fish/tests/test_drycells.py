import warnings

import numpy as np
import pytest
import xarray as xr

from pangeo_fish import helpers
from pangeo_fish.drycells import (
    DryCellWarning,
    count_dry_cells,
    dry_mask,
    mask_dry_cells,
)

TIMES = np.array(
    ["2025-03-31T00:00", "2025-03-31T01:00", "2025-03-31T02:00"],
    dtype="datetime64[ns]",
)


def make_model(H0=(np.nan, 10.0, 1.0, 0.5), XE=None):
    """1 x 4 cells (land, deep, shallow, very shallow), 3 hourly steps."""
    H0 = np.array(H0, dtype="float64")[None, :]
    if XE is None:
        XE = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, -2.0, -2.0, -2.0], [0.0, -1.0, -1.0, -0.5]]
        )
    XE = np.asarray(XE, dtype="float64")[:, None, :]
    depth = np.array([0.0, 3.0, 5.0])
    temp = np.where(np.isnan(H0), np.nan, 12.5)[None, None, :, :] + np.zeros(
        (3, 3, 1, 4)
    )
    ds = xr.Dataset(
        {
            "TEMP": (("time", "depth", "lat", "lon"), temp, {"units": "degC"}),
            "H0": (("lat", "lon"), H0),
            "XE": (("time", "lat", "lon"), XE),
        },
        coords={
            "time": TIMES,
            "depth": depth,
            "lat": [48.5],
            "lon": [-2.0, -1.9, -1.8, -1.7],
        },
    )
    ds["time"].attrs["standard_name"] = "time"
    ds["dynamic_depth"] = (ds["depth"] + ds["XE"]).where(ds["H0"].notnull())
    ds["dynamic_bathymetry"] = ds["H0"] + ds["XE"]
    return ds


def make_tag():
    time = np.arange(
        TIMES[0], TIMES[-1] + np.timedelta64(1, "h"), np.timedelta64(10, "m")
    )
    return xr.Dataset(
        {
            "temperature": ("time", np.full(time.size, 12.0)),
            "pressure": ("time", np.where(np.arange(time.size) % 2, 2.0, 0.5)),
        },
        coords={"time": time},
    )


def run_compute_diff(model, **kwargs):
    diff, _ = helpers.compute_diff(
        reference_model=model,
        tag_log=make_tag(),
        relative_depth_threshold=0,
        chunk_time=1,
        **kwargs,
    )
    return diff.compute()


def test_dry_mask_threshold_and_land():
    model = make_model()
    mask = dry_mask(model)
    assert mask.dims == ("time", "lat", "lon")
    expected = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
    np.testing.assert_array_equal(mask.values[:, 0, :], expected)
    np.testing.assert_array_equal(dry_mask(model, 0.6).values[0, 0, :], [0, 0, 0, 1])

    model["XE"][0, 0, 3] = np.nan
    assert not dry_mask(model, 10.0).values[0, 0, 3]


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf, "x", True])
def test_invalid_min_water_depth(value):
    with pytest.raises(ValueError, match="min_water_depth"):
        dry_mask(make_model(), value)


def test_count_dry_cells():
    counts = count_dry_cells(make_model())
    assert int(counts["n_sea"]) == 3
    np.testing.assert_array_equal(counts["n_dry"], [0, 2, 2])
    np.testing.assert_allclose(counts["min_water_column"], [0.5, -1.5, 0.0])
    np.testing.assert_array_equal(counts["n_top_level_above_surface"], [0, 1, 1])


def test_mask_dry_cells_only_touches_dry_pairs():
    model = make_model()
    ds = model[["H0", "XE"]].assign(diff=(("time", "lat", "lon"), np.ones((3, 1, 4))))
    ds["diff"].attrs["units"] = "degC"
    masked = mask_dry_cells(ds)
    dry = dry_mask(ds).values
    assert np.isnan(masked["diff"].values[dry]).all()
    np.testing.assert_array_equal(masked["diff"].values[~dry], ds["diff"].values[~dry])
    assert masked["diff"].dtype == ds["diff"].dtype
    assert masked["diff"].attrs == ds["diff"].attrs
    xr.testing.assert_identical(masked[["H0", "XE"]], ds[["H0", "XE"]])


def test_mask_dry_cells_keeps_time_steps_where_every_sea_cell_is_dry():
    XE = [[0.0] * 4, [0.0, -20.0, -2.0, -2.0], [0.0] * 4]
    ds = make_model(XE=XE)[["H0", "XE"]]
    ds = ds.assign(diff=(("time", "lat", "lon"), np.ones((3, 1, 4))))
    masked = mask_dry_cells(ds)
    np.testing.assert_array_equal(masked["diff"].values[1], ds["diff"].values[1])


def test_mask_dry_cells_is_lazy_and_works_on_healpix_cells():
    model = make_model()
    ds = model[["H0", "XE"]].assign(diff=(("time", "lat", "lon"), np.ones((3, 1, 4))))
    cells = ds.stack(cells=("lat", "lon")).drop_vars(["cells", "lat", "lon"])
    lazy = mask_dry_cells(cells.chunk({"time": 1}))
    assert lazy["diff"].chunks is not None
    xr.testing.assert_identical(lazy.compute(), mask_dry_cells(cells))


def test_compute_diff_none_keeps_dry_cells_and_warns():
    model = make_model()
    with pytest.warns(DryCellWarning, match="4 sea"):
        diff = run_compute_diff(model, min_water_depth=None)
    assert "min_water_depth" not in diff.attrs
    assert np.isfinite(diff["diff"].values[1:, 0, 2:]).all()


def test_compute_diff_default_masks_dry_cells_automatically():
    model = make_model()
    with pytest.warns(DryCellWarning, match="automatic masking"):
        auto = run_compute_diff(model)
    with pytest.warns(DryCellWarning):
        explicit = run_compute_diff(model, min_water_depth=0.0)
    xr.testing.assert_identical(auto, explicit)


def test_compute_diff_default_is_untouched_without_dry_cells():
    model = make_model(H0=(np.nan, 10.0, 5.0, 6.0), XE=np.full((3, 4), -0.9))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DryCellWarning)
        auto = run_compute_diff(model)
        none = run_compute_diff(model, min_water_depth=None)
    xr.testing.assert_identical(auto, none)
    assert "min_water_depth" not in auto.attrs


def test_compute_diff_min_water_depth_masks_dry_cells():
    model = make_model()
    with pytest.warns(DryCellWarning, match="Masking up to 4"):
        masked = run_compute_diff(model, min_water_depth=0.0)
    with pytest.warns(DryCellWarning):
        default = run_compute_diff(model, min_water_depth=None)
    dry = dry_mask(model).values
    assert np.isnan(masked["diff"].values[dry]).all()
    np.testing.assert_array_equal(
        masked["diff"].values[~dry], default["diff"].values[~dry]
    )
    assert masked.attrs["min_water_depth"] == 0.0
    for name in ["H0", "XE", "ocean_mask"]:
        xr.testing.assert_identical(masked[name], default[name])


def test_compute_diff_time_step_with_every_sea_cell_dry_is_left_unmasked():
    model = make_model(
        XE=[[0.0] * 4, [0.0, -20.0, -2.0, -2.0], [0.0, -1.0, -1.0, -0.5]]
    )
    with pytest.warns(
        DryCellWarning,
        match=r"Every sea cell is dry at 1 time steps \(2025-03-31T01:00\)",
    ):
        diff = run_compute_diff(model, min_water_depth=0.0)
    np.testing.assert_array_equal(diff["diff"].count(["lat", "lon"]), [3, 3, 1])


def test_compute_diff_min_water_depth_is_a_no_op_without_dry_cells():
    model = make_model(H0=(np.nan, 10.0, 5.0, 6.0), XE=np.full((3, 4), -0.9))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DryCellWarning)
        default = run_compute_diff(model, min_water_depth=None)
        masked = run_compute_diff(model, min_water_depth=0.0)
    assert default["diff"].values.tobytes() == masked["diff"].values.tobytes()
    assert masked.attrs["min_water_depth"] == 0.0


def test_dry_cell_warning_survives_runtime_warning_filter():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        run_compute_diff(make_model())
    assert any(issubclass(w.category, DryCellWarning) for w in record)


def test_compute_diff_data_vars_unchanged():
    model = make_model()
    with pytest.warns(DryCellWarning):
        diff = run_compute_diff(model)
    assert set(diff.data_vars) == {"diff", "H0", "XE", "ocean_mask"}


def test_all_dry_guard_ignores_sea_cells_without_valid_diff():
    # cell 1 is sea (finite H0) but has no valid diff; cells 2 and 3 are dry at t=1
    ds = make_model()[["H0", "XE"]]
    values = np.ones((3, 1, 4))
    values[:, :, 1] = np.nan
    ds = ds.assign(diff=(("time", "lat", "lon"), values))
    masked = mask_dry_cells(ds)
    assert np.isfinite(masked["diff"].values[1, 0, 2:]).all()


def test_compute_diff_with_extra_dimension():
    model = make_model()
    H0 = model["H0"]
    model = xr.concat([model, model], dim="member")
    model["H0"] = H0
    with pytest.warns(DryCellWarning, match="at most 2 of 3 sea cells"):
        diff = run_compute_diff(model, min_water_depth=None)
    with pytest.warns(DryCellWarning, match="Masking up to 8"):
        masked = run_compute_diff(model, min_water_depth=0.0)
    assert int(masked["diff"].isnull().sum() - diff["diff"].isnull().sum()) == 8


def test_compute_diff_default_stays_lazy():
    import dask

    calls = []

    class Counter(dask.callbacks.Callback):
        def _start(self, dsk):
            calls.append(
                {str(key[0] if isinstance(key, tuple) else key) for key in dsk}
            )

    model = make_model().chunk({"time": 1})
    with Counter(), pytest.warns(DryCellWarning):  # default "auto" masks lazily
        diff, _ = helpers.compute_diff(
            reference_model=model,
            tag_log=make_tag(),
            relative_depth_threshold=0,
            chunk_time=1,
        )
    assert diff["diff"].chunks is not None
    # only the dry-cell count (H0 and XE) is computed, not the diff
    assert len(calls) == 1
    assert not any("diff_z" in key for key in calls[0])


def test_time_step_with_missing_tag_data_is_not_reported_as_all_dry():
    ds = make_model()[["H0", "XE"]]
    values = np.ones((3, 1, 4))
    values[2] = np.nan  # no tag data at t=2, where cells 2 and 3 are dry
    ds = ds.assign(diff=(("time", "lat", "lon"), values))
    with pytest.warns(DryCellWarning) as record:
        masked = _apply(ds)
    assert "Every sea cell is dry" not in str(record[0].message)
    np.testing.assert_array_equal(
        np.isfinite(masked["diff"].values[:, 0, :]),
        [[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]],
    )


def _apply(ds):
    from pangeo_fish.drycells import _apply_min_water_depth

    return _apply_min_water_depth(ds, 0.0)
