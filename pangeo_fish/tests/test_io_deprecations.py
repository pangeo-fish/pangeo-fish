import sys
import types

import pytest


def test_open_copernicus_zarr_is_deprecated(monkeypatch):
    from pangeo_fish.io import open_copernicus_zarr

    class Client:
        @staticmethod
        def open(url):
            raise ConnectionError("no network in tests")

    monkeypatch.setitem(
        sys.modules, "pystac_client", types.SimpleNamespace(Client=Client)
    )
    with pytest.warns(FutureWarning, match="load_model"):
        with pytest.raises(ConnectionError):
            open_copernicus_zarr()
