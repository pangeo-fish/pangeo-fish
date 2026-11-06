from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from pangeo_fish import helpers


@pytest.fixture
def coastal_source():
    lon, lat = np.meshgrid([-8.0, -7.0, -6.0], [48.0, 49.0])
    return xr.Dataset(
        {
            "temperature": (("time", "yi", "xi"), np.arange(12.0).reshape(2, 2, 3)),
            "depth": (("yi", "xi"), np.arange(6.0).reshape(2, 3)),
            "ocean_mask": (("yi", "xi"), [[True, True, False], [True, True, False]]),
        },
        coords={
            "time": [0, 1],
            "longitude": (("yi", "xi"), lon),
            "latitude": (("yi", "xi"), lat),
        },
    )


def test_regrid_fills_only_missing_ocean_values(monkeypatch, coastal_source):
    """Permuted output IDs must align, and CT values and the land mask must survive."""
    calls = []

    class Bilinear:
        def __init__(self, *, threshold, **kwargs):
            self.is_mask = threshold == -1

        def get_cell_ids(self):
            return np.array([9, 2, 7, 4, 1] if self.is_mask else [4, 9, 1, 7])

        def resample(self, values):
            if self.is_mask:
                data = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
            else:
                calls.append(values.copy())
                data = (
                    np.array([[40.0, 900.0, 10.0, 700.0], [41.0, 901.0, 11.0, 701.0]])
                    if values.ndim == 2
                    else np.array([40.0, 900.0, 10.0, 700.0])
                )
            return SimpleNamespace(cell_data=data)

    class CloughTocher:
        def __init__(self, **kwargs):
            pass

        def get_cell_ids(self):
            return np.array([7, 1, 9])

        def resample(self, values):
            data = (
                np.array([[70.0, 1.0, 90.0], [71.0, np.nan, 91.0]])
                if values.ndim == 2
                else np.array([70.0, 1.0, 90.0])
            )
            return SimpleNamespace(cell_data=data)

    monkeypatch.setattr(helpers, "BilinearResampler", Bilinear)
    monkeypatch.setattr(helpers, "CloughTocherResampler", CloughTocher)
    result, _ = helpers.regrid_dataset_hpresample(ds=coastal_source, refinement_level=0)

    np.testing.assert_array_equal(result.cell_ids, [1, 2, 4, 7, 9])
    np.testing.assert_allclose(
        result.temperature, [[1, np.nan, 40, 70, np.nan], [11, np.nan, 41, 71, np.nan]]
    )
    np.testing.assert_allclose(result.depth, [1, np.nan, 40, 70, np.nan])
    np.testing.assert_array_equal(result.ocean_mask, [1, 1, 1, 1, 0])
    # Both time-dependent and static fallback calls use ocean source points only.
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], [[0, 1, 3, 4], [6, 7, 9, 10]])
    np.testing.assert_array_equal(calls[1], [0, 1, 3, 4])


def test_regrid_coastal_constant_with_real_resamplers():
    """Ocean cells covered only by bilinear interpolation retain the constant field."""
    lon, lat = np.meshgrid(np.arange(-8.0, -3.0, 0.1), np.arange(47.0, 51.0, 0.1))
    sea = lon < -5.5
    source = xr.Dataset(
        {
            "temperature": (("time", "yi", "xi"), np.where(sea, 17.0, np.nan)[None]),
            "ocean_mask": (("yi", "xi"), sea),
        },
        coords={
            "time": [0],
            "longitude": (("yi", "xi"), lon),
            "latitude": (("yi", "xi"), lat),
        },
    )
    ct = helpers.CloughTocherResampler(
        lon_deg=lon[sea], lat_deg=lat[sea], level=8, ellipsoid="sphere"
    )
    bil = helpers.BilinearResampler(
        lon_deg=lon[sea],
        lat_deg=lat[sea],
        level=8,
        ellipsoid="sphere",
        threshold=0.5,
        ring_search_max=4,
    )
    result, _ = helpers.regrid_dataset_hpresample(ds=source, refinement_level=8)
    ocean = result.ocean_mask.values.astype(bool)
    fallback = (
        ocean
        & ~np.isin(result.cell_ids, ct.get_cell_ids())
        & np.isin(result.cell_ids, bil.get_cell_ids())
    )
    unsupported = (
        ocean
        & ~np.isin(result.cell_ids, ct.get_cell_ids())
        & ~np.isin(result.cell_ids, bil.get_cell_ids())
    )
    assert fallback.any()
    np.testing.assert_allclose(result.temperature.values[:, fallback], 17.0)
    assert np.isnan(result.temperature.values[:, ~ocean]).all()
    assert np.isnan(result.temperature.values[:, unsupported]).all()
