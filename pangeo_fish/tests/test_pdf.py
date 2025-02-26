import numpy as np
import xarray as xr

from pangeo_fish.pdf import combine_emission_pdf


def test_single_emission_with_final():
    """
    One emission variable ("em") with a "final"
    """
    times = np.arange(3)
    latitudes = np.array([10, 20])
    longitudes = np.array([100])

    em = xr.DataArray(
        np.full((3, len(latitudes), len(longitudes)), 2.0),
        dims=("time", "latitude", "longitude"),
        name="pdf",
    )

    # "Final"
    final = xr.DataArray(
        np.full((len(latitudes), len(longitudes)), 3.0),
        dims=("latitude", "longitude"),
        name="final",
    )

    initial = xr.DataArray(np.array([1]), dims=("dummy",), name="initial")
    mask = xr.DataArray(np.array([0]), dims=("dummy",), name="mask")

    # Create the dataset
    ds = xr.Dataset(
        {"em": em, "final": final, "initial": initial, "mask": mask},
        coords={"latitude": latitudes, "longitude": longitudes, "time": times},
    )

    result = combine_emission_pdf(ds)

    assert all(var in result for var in {"pdf", "initial", "final", "mask"})

    # Time 0 and 1: sum is 4 → pdf = 2/4 = 0.5
    # Time 2: sum is 12 → pdf = 6/12 = 0.5
    expected = xr.full_like(em, fill_value=0.5).assign_coords(ds.coords)

    xr.testing.assert_allclose(result["pdf"], expected, rtol=1e-7, atol=1e-7)


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

    # Normalized pdf calculation:
    # Time 0: em = [[3], [1]] → sum = 3+1 = 4 → pdf = [[3/4], [1/4]] = [[0.75], [0.25]]
    # Time 1: em = [[2], [2]] → sum = 2+2 = 4 → pdf = [[0.5], [0.5]]
    expected = ds.assign(
        pdf=em.copy(data=np.array([[[0.75], [0.25]], [[0.5], [0.5]]]))
    ).drop_vars("em")
    xr.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
