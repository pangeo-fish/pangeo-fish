"""Publishing must preserve probabilities, zero-valued IDs and parent tiles."""

import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.storage import MemoryStore

from pangeo_fish.storage import prepare_dggs_zarr, write_dggs_zarr


@pytest.fixture
def sparse_maps():
    ds = xr.Dataset(
        {
            "states": (("cells", "time"), [[0.0, 0.2], [0.3, np.nan], [0.7, 0.8]]),
            "mask": ("cells", [True, False, True]),
            "sigma": ("time", [0.1, 0.2]),
        },
        coords={
            "cell_ids": ("cells", np.array([0, 2, 19], dtype="uint64")),
            "time": np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"),
        },
        attrs={"title": "Fish posterior", "zarr_conventions": [{"name": "other"}]},
    )
    ds.cell_ids.attrs.update(
        grid_name="healpix", level=2, indexing_scheme="nested", ellipsoid="WGS84"
    )
    ds.states.attrs["units"] = "1"
    return ds


def test_sparse_metadata_and_no_mutation(sparse_maps):
    source = sparse_maps.copy(deep=True)
    sparse_maps.states.encoding = {"chunks": (3, 2), "_FillValue": 0.0}
    ds = prepare_dggs_zarr(sparse_maps.chunk({"time": 2, "cells": 1}))
    assert ds.states.dims == ("time", "cells")
    assert ds.states.chunks == ((1, 1), (3,))
    assert ds.attrs["dggs"] == {
        "name": "healpix",
        "refinement_level": 2,
        "indexing_scheme": "nested",
        "ellipsoid": {
            "name": "WGS84",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
        "spatial_dimension": "cells",
        "coordinate": "cell_ids",
        "compression": "none",
    }
    assert len(prepare_dggs_zarr(ds).attrs["zarr_conventions"]) == 2
    xr.testing.assert_identical(sparse_maps, source)
    assert sparse_maps.states.encoding["_FillValue"] == 0.0


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize(
    "ellipsoid", [None, "sphere", "WGS84", {"name": "test", "radius": 1234.0}]
)
def test_parent_chunks_roundtrip(tmp_path, sparse_maps, indexed, ellipsoid):
    import xdggs  # noqa: F401

    if ellipsoid is None:
        del sparse_maps.cell_ids.attrs["ellipsoid"]
    else:
        sparse_maps.cell_ids.attrs["ellipsoid"] = ellipsoid
    if indexed:
        sparse_maps = sparse_maps.dggs.decode()
    source = sparse_maps.copy(deep=True)
    path = tmp_path / "states.zarr"
    exported = write_dggs_zarr(sparse_maps, path, chunk_level=1)
    actual = xr.open_zarr(path).load()
    # Only occupied parents 0 and 4, each with four children. No global padding.
    np.testing.assert_array_equal(actual.cell_ids, [0, 1, 2, 3, 16, 17, 18, 19])
    selected = actual.swap_dims(cells="cell_ids").sel(cell_ids=[0, 2, 19])
    np.testing.assert_allclose(
        selected.states.values, source.states.transpose("time", "cells")
    )
    assert actual.states.isel(cells=[1, 3, 4, 5, 6]).isnull().all()
    assert not actual["mask"].isel(cells=[1, 3, 4, 5, 6]).any()
    assert actual.states[0, 0] == 0.0
    assert actual.cell_ids[0] == 0
    xr.testing.assert_identical(actual.sigma, source.sigma)
    assert actual.states.attrs["units"] == "1"
    group = zarr.open_group(path)
    assert group.metadata.zarr_format == 2
    assert group["states"].chunks == (1, 4)
    assert group["cell_ids"].chunks == (4,)
    assert group["states"].compressors[0].cname == "lz4"
    assert (path / ".zmetadata").is_file()
    xr.testing.assert_identical(actual, exported.compute())
    xr.testing.assert_identical(sparse_maps, source)
    assert actual.dggs.decode().dggs.grid_info.level == 2
    with pytest.raises(FileExistsError):
        write_dggs_zarr(sparse_maps, path, chunk_level=1)


def test_default_chunks_and_memory_store(sparse_maps):
    store = MemoryStore()
    ds = write_dggs_zarr(sparse_maps, store)
    # Level 2: one base-face tile, containing 4**2 children.
    assert zarr.open_group(store)["states"].chunks == (1, 16)
    assert ds.sizes["cells"] == 32
    assert xr.open_zarr(store).states[0, 0] == 0.0


@pytest.mark.parametrize(
    "bad_ids", [[0, 0, 1], [-1, 2, 3], [0, 2, 192], [0.0, 2.0, 3.0]]
)
def test_invalid_ids_fail_before_writing(tmp_path, sparse_maps, bad_ids):
    sparse_maps = sparse_maps.assign_coords(cell_ids=("cells", bad_ids))
    sparse_maps.cell_ids.attrs.update(
        grid_name="healpix", level=2, indexing_scheme="nested"
    )
    with pytest.raises(ValueError, match="cell_ids"):
        write_dggs_zarr(sparse_maps, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_ring_and_invalid_chunk_level(tmp_path, sparse_maps):
    sparse_maps.cell_ids.attrs["indexing_scheme"] = "ring"
    assert prepare_dggs_zarr(sparse_maps).attrs["dggs"]["indexing_scheme"] == "ring"
    with pytest.raises(ValueError, match="nested"):
        write_dggs_zarr(sparse_maps, tmp_path / "ring")
    sparse_maps.cell_ids.attrs["indexing_scheme"] = "nested"
    for value in [-1, 3, 0.5]:
        with pytest.raises(ValueError, match="chunk_level"):
            write_dggs_zarr(sparse_maps, tmp_path / "invalid", chunk_level=value)


def test_named_dataarray_and_unsorted_ids(tmp_path, sparse_maps):
    da = sparse_maps.states.isel(cells=[2, 0, 1])
    ds = prepare_dggs_zarr(da)
    assert list(ds.data_vars) == ["states"]
    np.testing.assert_array_equal(ds.cell_ids, [19, 0, 2])
    path = tmp_path / "dataarray.zarr"
    write_dggs_zarr(da, path, chunk_level=1)
    actual = xr.open_zarr(path)
    assert actual.states[0, 0] == 0.0
    assert actual.states[0, -1] == 0.7
