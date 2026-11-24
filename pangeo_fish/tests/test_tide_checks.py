"""Loud failures in the tide merge and inputs."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish import tide
from pangeo_fish import tide_healpix as th


def _write_stores(path, tide_cells, tide_times, total_cells, total_times):
    tide_ids = th.make_cell_ids(tide_cells, 2, "WGS84")
    pdf = np.tile(np.array([0.1, 0.3, 0.6])[: len(tide_cells)], (len(tide_times), 1))
    pdf = pdf / pdf.sum(axis=1, keepdims=True)
    likelihood = xr.Dataset(
        {
            "pdf": (("time", "cells"), pdf),
            "ocean_mask": ("cells", np.ones(len(tide_cells), bool)),
            "tide_found": ("time", np.ones(len(tide_times))),
        },
        coords={"time": tide_times, "cell_ids": tide_ids},
    )
    th.write_tide_pdf_healpix(likelihood, path, behav_found=[1] * len(tide_times))
    n = len(total_times)
    total = xr.Dataset(
        {
            "pdf": (
                ("time", "cells"),
                np.full((n, len(total_cells)), 1.0 / len(total_cells)),
            ),
            "initial": ("cells", np.arange(len(total_cells), dtype=float)),
            "final": ("cells", np.arange(len(total_cells), dtype=float)[::-1].copy()),
            "mask": ("time", np.arange(n) % 2),
        },
        coords={
            "time": total_times,
            "cell_ids": th.make_cell_ids(total_cells, 2, "WGS84"),
        },
        attrs={"sources": ["temperature"]},
    )
    total.to_zarr(path / "total_pdf.zarr", zarr_format=2, consolidated=True)
    return total


def test_merge_keeps_all_cells_and_days_with_neutral_fill(tmp_path):
    times = pd.date_range("2024-01-01", periods=4)
    total = _write_stores(tmp_path, [1, 2], times[:2], [1, 2, 3], times)
    with pytest.warns(UserWarning, match=r"1 of 3 cells.*2 of 4 days"):
        merged, tide_mask, behav_mask = tide.merge_tide_with_total(tmp_path)
    assert merged.sizes == total.sizes
    np.testing.assert_array_equal(merged.cell_ids, total.cell_ids)
    for name in ("pdf", "initial", "final", "mask"):
        np.testing.assert_array_equal(merged[name], total[name])
    p = merged["pdf_tide"].values
    # days with tide: known cells keep values, unknown cell gets the day's mean
    np.testing.assert_allclose(p[:2, :2], [[0.25, 0.75]] * 2)
    np.testing.assert_allclose(p[:2, 2], 0.5)
    # days without tide: uniform
    np.testing.assert_allclose(p[2:], 1 / 3)
    np.testing.assert_array_equal(tide_mask.isel(cells=0).values, [1, 1, 0, 0])
    np.testing.assert_array_equal(behav_mask.isel(cells=0).values, [1, 1, 0, 0])


def test_merge_without_gaps_does_not_warn(tmp_path, recwarn):
    times = pd.date_range("2024-01-01", periods=2)
    _write_stores(tmp_path, [1, 2], times, [2, 1], times)
    merged, _, _ = tide.merge_tide_with_total(tmp_path)
    assert not [w for w in recwarn if "neutral" in str(w.message)]
    np.testing.assert_allclose(merged.pdf_tide.values[0], [0.75, 0.25])


@pytest.fixture
def tpxo_like():
    lat = np.arange(50.0, 53.01, 1.0)
    lon = np.arange(358.0, 363.0, 1.0) % 360  # 358..2 in 0-360
    return xr.Dataset(
        {"h": (("lat", "lon"), np.zeros((lat.size, lon.size)))},
        coords={"lat": lat, "lon": lon},
    )


def test_bbox_outside_file_raises(tpxo_like):
    with pytest.raises(ValueError, match="outside the tide file coverage"):
        tide.convert_lon_360_to_180(
            tpxo_like, {"latitude": [46, 48], "longitude": [-1, 1]}
        )
    with pytest.raises(ValueError, match="longitude range"):
        tide.convert_lon_360_to_180(
            tpxo_like, {"latitude": [50, 52], "longitude": [-7, -5]}
        )


def test_bbox_partly_outside_warns_with_clipped_extent(tpxo_like):
    with pytest.warns(UserWarning, match=r"clipped to \[50.0, 53.0\]"):
        out = tide.convert_lon_360_to_180(
            tpxo_like, {"latitude": [46, 53], "longitude": [-1, 1]}
        )
    assert float(out.lat.min()) == 50.0 and float(out.lat.max()) == 53.0


def test_bbox_inside_file_is_silent(tpxo_like, recwarn):
    out = tide.convert_lon_360_to_180(
        tpxo_like, {"latitude": [51, 52], "longitude": [-1, 1]}
    )
    assert not recwarn.list
    assert out.lat.values.tolist() == [51.0, 52.0]
    assert out.lon.values.tolist() == [-1.0, 0.0, 1.0]


def _tag(values, units=None):
    time = pd.date_range("2024-01-01", periods=len(values), freq="10min")
    attrs = {} if units is None else {"units": units}
    return xr.Dataset({"pressure": ("time", values, attrs)}, coords={"time": time})


@pytest.mark.parametrize("units", ["dbar", "m", "DBAR", None])
def test_pressure_units_accepted(units):
    df = tide.prepare_pressure(_tag(np.linspace(5, 20, 30), units))
    assert len(df) == 30


@pytest.mark.parametrize("units", ["Pa", "hPa", "mbar"])
def test_pressure_wrong_units_attribute_raises(units):
    with pytest.raises(ValueError, match="needs? dbar or m|need dbar or m"):
        tide.prepare_pressure(_tag(np.linspace(5, 20, 30), units))


def test_pressure_in_pa_labelled_dbar_raises():
    with pytest.raises(ValueError, match="impossible for a depth"):
        tide.prepare_pressure(_tag(np.linspace(101325, 111325, 30), "dbar"))


def test_tide_behav_extr_checks_plausibility():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=50, freq="10min"),
            "pressure": np.full(50, 150000.0),
        }
    )
    with pytest.raises(ValueError, match="impossible for a depth"):
        tide.tide_behav_extr(df, "t")


def test_sigma_ignored_by_legacy_warns():
    with pytest.warns(UserWarning, match="ignored by tide_method='legacy'"):
        tide.check_tide_sigma("legacy", 1.0)


def test_sigma_ignored_by_pedersen_warns():
    with pytest.warns(UserWarning, match="ignored by tide_method='pedersen'"):
        tide.check_tide_sigma("pedersen", 1.0)


def test_legacy_method_warns_and_unknown_raises():
    with pytest.warns(UserWarning, match="incorrect"):
        tide.check_tide_method("legacy")
    with pytest.raises(ValueError, match="tide_method"):
        tide.check_tide_method("nope")


@pytest.mark.parametrize(
    ("method", "sigma"),
    [("legacy", 2.0), ("legacy", None), ("healpix", 1.0), ("pedersen", 2.0)],
)
def test_sigma_no_warning(method, sigma, recwarn):
    tide.check_tide_sigma(method, sigma)
    assert not recwarn.list


@pytest.mark.parametrize("method", ["pedersen", "healpix"])
def test_valid_methods_do_not_warn(method, recwarn):
    tide.check_tide_method(method)
    assert not recwarn.list
