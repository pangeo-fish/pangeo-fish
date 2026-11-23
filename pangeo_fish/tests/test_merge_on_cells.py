"""Variables are merged into the combined emission by cell id, not by position."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.helpers import merge_on_cells

GRID = {
    "grid_name": "healpix",
    "level": 1,
    "indexing_scheme": "nested",
    "ellipsoid": "WGS84",
}


def _ds(name, ids, values, times):
    return xr.Dataset(
        {name: (("time", "cells"), values)},
        coords={"cell_ids": ("cells", np.asarray(ids), GRID), "time": times},
    )


def test_cells_are_matched_by_id():
    times = pd.date_range("2024-01-01", periods=2)
    total = _ds("pdf", [3, 5, 7, 9], np.ones((2, 4)), times)
    total.attrs["sources"] = ["temperature"]
    # other in another order, missing cell 7, with an extra cell 11
    other = _ds("pdf_bathy", [9, 3, 11, 5], [[0.9, 0.3, 1.1, 0.5]] * 2, times)
    other["pdf_bathy"].attrs["units"] = "1"

    merged = merge_on_cells(total, other)
    np.testing.assert_array_equal(merged.cell_ids, [3, 5, 7, 9])
    np.testing.assert_allclose(merged.pdf_bathy[0], [0.3, 0.5, np.nan, 0.9])
    assert merged.pdf_bathy.attrs["units"] == "1"
    assert merged.attrs["sources"] == ["temperature"]
    assert merged.cell_ids.attrs == GRID


def test_time_steps_are_matched_by_label():
    times = pd.date_range("2024-01-01", periods=3)
    total = _ds("pdf", [1, 2], np.ones((3, 2)), times)
    other = _ds("acoustic", [1, 2], [[2.0, 2.0], [3.0, 3.0]], times[[2, 0]])
    merged = merge_on_cells(total, other)
    np.testing.assert_allclose(merged.acoustic[:, 0], [3.0, np.nan, 2.0])


def test_other_grid_is_refused():
    times = pd.date_range("2024-01-01", periods=1)
    total = _ds("pdf", [1, 2], np.ones((1, 2)), times)
    other = _ds("pdf_bathy", [1, 2], np.ones((1, 2)), times)
    other["cell_ids"].attrs = GRID | {"ellipsoid": "sphere"}
    with pytest.raises(ValueError, match="different HEALPix grids"):
        merge_on_cells(total, other)
