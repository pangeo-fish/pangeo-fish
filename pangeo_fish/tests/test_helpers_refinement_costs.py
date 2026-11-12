import numpy as np
import pytest
import xarray as xr

from pangeo_fish import helpers

BBOX = {"latitude": (50.7, 53.0), "longitude": (1.5, 5.08), "max_depth": 50}
PERIOD = slice("2023-05-03", "2023-06-08")


def bbox_area_km2(bbox, radius=6371.0):
    (lon0, lon1), (lat0, lat1) = bbox["longitude"], bbox["latitude"]
    return (
        radius**2
        * np.deg2rad(lon1 - lon0)
        * (np.sin(np.deg2rad(lat1)) - np.sin(np.deg2rad(lat0)))
    )


def test_table_without_mask():
    table = helpers.estimate_refinement_costs(bbox=BBOX, time_slice=PERIOD)

    assert list(table["refinement_level"]) == [8, 9, 10, 11, 12, 13]
    assert (table["n_steps"] == 37).all()  # both ends included
    assert table["ocean_share"].isna().all()
    assert table["n_ocean_cells"].isna().all()

    # cell size: sqrt of the cell area on a 6371 km sphere (~6.4 km at level 10)
    level10 = table.set_index("refinement_level").loc[10]
    assert level10["cell_size_km"] == pytest.approx(6.371, abs=0.01)
    assert np.allclose(
        table["cell_size_km"].values[:-1] / table["cell_size_km"].values[1:], 2
    )

    # cells covering the bbox ~ bbox area / cell area (plus border cells)
    fine = table.set_index("refinement_level").loc[13]
    expected = bbox_area_km2(BBOX) / fine["cell_size_km"] ** 2
    assert fine["n_cells"] == pytest.approx(expected, rel=0.05)
    assert (np.diff(table["n_cells"]) > 0).all()

    # memory: cells x steps x 8 bytes, 5 arrays by default
    assert np.allclose(table["array_mb"], table["n_cells"] * 37 * 8 / 1e6)
    assert np.allclose(table["memory_mb"], 5 * table["array_mb"])


def test_reference_scaling():
    table = helpers.estimate_refinement_costs(
        bbox=BBOX, n_steps=36, refinement_levels=[11]
    )
    row = table.iloc[0]
    # A19845 reference: 6,467 cells after regridding vs. cells covering the bbox
    assert row["n_cells"] == pytest.approx(6467, rel=0.02)
    assert row["relative_cost"] == pytest.approx(
        row["n_cells"] / helpers.REFERENCE_RUN["n_cells"]
    )

    reference = {"n_cells": int(row["n_cells"]), "n_steps": 36, "minutes": 10.0}
    table = helpers.estimate_refinement_costs(
        bbox=BBOX,
        n_steps=72,
        refinement_levels=[11, 12],
        reference=reference,
        n_arrays=7,
    ).set_index("refinement_level")
    assert table.loc[11, "relative_cost"] == pytest.approx(2.0)
    assert table.loc[11, "runtime_minutes"] == pytest.approx(20.0)
    assert table.loc[11, "memory_mb"] == pytest.approx(7 * table.loc[11, "array_mb"])
    assert table.loc[12, "relative_cost"] / table.loc[11, "relative_cost"] == (
        pytest.approx(4, rel=0.05)
    )


@pytest.mark.parametrize("names", [("lat", "lon"), ("latitude", "longitude")])
def test_ocean_mask(names):
    lat = np.arange(50.0, 54.0, 0.02)
    lon = np.arange(1.0, 6.0, 0.02)
    # ocean west of 3.29 E (the bbox mid-longitude), land east of it
    values = np.broadcast_to(lon < 3.29, (lat.size, lon.size))
    mask = xr.DataArray(
        values, dims=names, coords={names[0]: lat, names[1]: lon}
    ).astype(float)
    mask[0, 0] = np.nan  # NaN counts as land

    table = helpers.estimate_refinement_costs(
        bbox=BBOX, n_steps=10, refinement_levels=[10, 12], ocean_mask=mask
    ).set_index("refinement_level")
    for level in (10, 12):
        row = table.loc[level]
        assert row["ocean_share"] == pytest.approx(0.5, abs=0.05)
        assert row["n_ocean_cells"] == round(row["ocean_share"] * row["n_cells"])
        # arrays keep land cells, so memory follows all cells
        assert row["array_mb"] == pytest.approx(row["n_cells"] * 10 * 8 / 1e6)


def test_ellipsoid():
    sphere = helpers.estimate_refinement_costs(
        bbox=BBOX, n_steps=1, refinement_levels=[10]
    )
    wgs84 = helpers.estimate_refinement_costs(
        bbox=BBOX, n_steps=1, refinement_levels=[10], ellipsoid="WGS84"
    )
    assert wgs84["cell_size_km"].item() > sphere["cell_size_km"].item()
    assert wgs84["n_cells"].item() == pytest.approx(sphere["n_cells"].item(), rel=0.05)


def test_requires_period():
    with pytest.raises(ValueError, match="time_slice"):
        helpers.estimate_refinement_costs(bbox=BBOX)
