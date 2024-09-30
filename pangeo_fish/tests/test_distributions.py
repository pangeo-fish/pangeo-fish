import numpy as np
import pytest
import xarray as xr
import xdggs  # noqa: F401
from healpix_convolution.distances import _distances
from healpix_convolution.tests.test_kernels import fit_polynomial

from pangeo_fish import distributions


class TestHealpix:
    @pytest.mark.parametrize(
        ["grid", "grid_info"],
        (
            (
                xr.Dataset(coords={"cell_ids": ("cells", np.arange(12 * 4**2))}),
                {"grid_name": "healpix", "resolution": 2, "indexing_scheme": "ring"},
            ),
            (
                xr.Dataset(
                    coords={"cell_ids": ("cells", np.arange(0 * 4**3, 2 * 4**3))}
                ),
                {"grid_name": "healpix", "resolution": 3, "indexing_scheme": "nested"},
            ),
        ),
    )
    @pytest.mark.parametrize(
        "pos",
        (
            xr.Dataset({"longitude": 1, "latitude": 50}),
            xr.Dataset({"longitude": 10, "latitude": 60}),
        ),
    )
    @pytest.mark.parametrize("sigma", (1e-2, 1, 5))
    def test_normal_at(self, grid, grid_info, pos, sigma):
        grid_ = grid.dggs.decode(grid_info=grid_info)
        grid_info_ = grid_.dggs.grid_info

        pdf = distributions.healpix.normal_at(grid_, pos, sigma).dggs.decode(
            grid_info=grid_info
        )

        lon = pos["longitude"].data
        lat = pos["latitude"].data
        expected_center = grid_info_.geographic2cell_ids(lon=lon, lat=lat)
        cell_ids = grid["cell_ids"].data

        center = pdf.dggs.cell_centers().reset_coords().weighted(pdf.fillna(0)).mean()
        print(center)
        print(pos)

        distances = _distances(
            np.reshape(expected_center, (1, 1)),
            np.reshape(cell_ids, (1, -1)),
            axis=-1,
            nside=grid_info_.nside,
            nest=grid_info_.nest,
        )

        assert (pdf > 0).sum() > 5
        x = np.squeeze(distances)
        y = pdf.fillna(0).data
        polynomial = fit_polynomial(x, y, deg=2)
        actual_sigma = np.sqrt(-1 / 2 / polynomial.coef[2])

        np.testing.assert_allclose(actual_sigma, sigma)
