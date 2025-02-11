import numpy as np
import pytest
import xarray as xr

from pangeo_fish.pdf import combine_emission_pdf


@pytest.fixture
def create_emission_dataset():
    """
    Fixture to create an emission dataset for testing.
    """
    times = np.arange(3)
    latitudes = np.array([10, 20])
    longitudes = np.array([100])

    em = xr.DataArray(
        np.full((3, len(latitudes), len(longitudes)), 2.0),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        name="pdf",
    )

    # "Final"
    final = xr.DataArray(
        np.full((len(latitudes), len(longitudes)), 3.0),
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes},
        name="final",
    )

    initial = xr.DataArray(np.array([1]), dims=("dummy",), name="initial")
    mask = xr.DataArray(np.array([0]), dims=("dummy",), name="mask")

    # Create the dataset
    ds = xr.Dataset({"em": em, "final": final, "initial": initial, "mask": mask})

    return ds


def test_single_emission_with_final(create_emission_dataset):
    """
    One emission variable ("em") with a "final"
    """
    ds = create_emission_dataset
    result = combine_emission_pdf(ds)

    for var in ["pdf", "initial", "final", "mask"]:
        assert var in result

    # Time 0 and 1: sum is 4 → pdf = 2/4 = 0.5
    # Time 2: sum is 12 → pdf = 6/12 = 0.5
    expected = np.full((3, len(ds.latitude), len(ds.longitude)), 0.5)

    np.testing.assert_allclose(result["pdf"].values, expected, rtol=1e-2, atol=1e-2)


def test_single_emission_without_final():
    """
    emission variable without final
    """
    times = np.arange(2)
    latitudes = np.array([0, 1])
    longitudes = np.array([100])

    em = xr.DataArray(
        np.array([[[3], [1]], [[2], [2]]]),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        name="em",
    )
    initial = xr.DataArray(np.array([5]), dims=("dummy",), name="initial")

    ds = xr.Dataset({"em": em, "initial": initial})
    result = combine_emission_pdf(ds)

    assert "pdf" in result
    assert "initial" in result
    assert "final" not in result

    # Normalized pdf calculation:
    # Time 0: em = [[3], [1]] → sum = 3+1 = 4 → pdf = [[3/4], [1/4]] = [[0.75], [0.25]]
    # Time 1: em = [[2], [2]] → sum = 2+2 = 4 → pdf = [[0.5], [0.5]]
    expected = np.array([[[0.75], [0.25]], [[0.5], [0.5]]])
    np.testing.assert_allclose(result["pdf"].values, expected, rtol=1e-7, atol=1e-7)
