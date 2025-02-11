import numpy as np
import pytest
import xarray as xr

from pangeo_fish.grid import center_longitude


@pytest.fixture
def sample_dataset():
    num_cells = 10  # Small grid size for testing

    # Generate longitudes
    lons = np.linspace(-200, 200, num_cells) % 360
    lons[lons < 0] += 360
    lats = np.linspace(-90, 90, num_cells)  # Latitudes

    ds = xr.Dataset(
        coords={
            "longitude": ("cells", lons),
            "latitude": ("cells", lats),
        }
    )
    return ds


def test_center_longitude_center_0(sample_dataset):
    ds = sample_dataset
    centered_ds = center_longitude(ds, center=0)

    # Check that longitudes are centered around 0 (-180 to 180)
    assert np.all(
        (centered_ds.longitude >= -180) & (centered_ds.longitude <= 180)
    ), "Longitudes are not properly centered around 0."
