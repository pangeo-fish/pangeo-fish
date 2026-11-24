"""Regression tests for the mixed-level water graph in the public workflow."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from healpix_analyse._topology import nested_neighbours
from pint_xarray import unit_registry as ureg

from pangeo_fish import helpers

pytestmark = pytest.mark.filterwarnings(
    "ignore:A .predictor_index. entry:RuntimeWarning"
)


@pytest.fixture
def emission():
    parent = 4 * 4**10 + 128
    children = 4 * parent + np.arange(4)
    neighbors = nested_neighbours(children, 11, connectivity="edge")
    for i, (source, source_neighbors) in enumerate(zip(children, neighbors)):
        bays = set(source_neighbors[source_neighbors >= 0] >> 2) - {parent}
        for j, (bridge, bridge_neighbors) in enumerate(zip(children, neighbors)):
            if bridge not in source_neighbors:
                continue
            mouths = set(bridge_neighbors[bridge_neighbors >= 0] >> 2) - {parent}
            for mouth in sorted(mouths):
                other_bays = sorted(bays - {mouth})
                if not other_bays:
                    continue
                land = children[next(k for k in range(4) if k not in (i, j))]
                levels = np.array([11, 11, 10, 10, 11])
                ids = np.array([source, bridge, mouth, other_bays[0], land])
                uniq = ids + (4 << (2 * levels))
                likelihood = np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.2, 1.0, 0.8, 2.0, 0.0],
                        [0.1, 0.3, 1.0, 4.0, 0.0],
                    ]
                )
                return xr.Dataset(
                    {
                        "pdf": (("time", "cells"), likelihood),
                        "mask": ("cells", [True, True, True, True, False]),
                        "initial": ("cells", [1.0, 0.0, 0.0, 0.0, 0.0]),
                        "wet_fraction": ("cells", [0.2, 0.25, 1.0, 1.0, 1.0]),
                        "water_contact_from": ("water_contacts", uniq[[0, 1]]),
                        "water_contact_to": ("water_contacts", uniq[[1, 2]]),
                    },
                    coords={
                        "time": pd.date_range("2022-01-01", periods=3, freq="h"),
                        "cell_ids": ("cells", ids, {"ellipsoid": "WGS84"}),
                        "cell_level": ("cells", levels),
                    },
                )
    raise AssertionError("failed to build river fixture")


def test_mixed_healpix_workflow_preserves_water_contacts(emission, tmp_path):
    params, prepared = helpers.optimize_pdf(
        ds=emission,
        conv_method="HealpixAnalyseMixedConv",
        adjustment_factor=1.0,
        truncate=4.0,
        maximum_speed=ureg.Quantity(20, "km/h"),
        tolerance=1e-6,
        save_parameters=True,
        target_root=str(tmp_path),
    )
    assert np.isfinite(params["sigma"]).all()
    saved = json.loads((tmp_path / "parameters.json").read_text())
    assert "GaussianMixedHealpixAnalyse" in saved["predictor_factory"]["class"]
    assert "water_contacts" not in saved["predictor_factory"]["kwargs"]

    prepared.to_zarr(tmp_path / "combined.zarr", zarr_format=2)
    states = helpers.compute_states(target_root=str(tmp_path), chunks={"time": 1})
    assert (tmp_path / "states.zarr").exists()
    assert states["cell_level"].values.tolist() == [11, 11, 10, 10, 11]
    mass = states["states"].fillna(0).values
    assert mass[-1, 2] > 0
    assert np.count_nonzero(mass[:, 3:]) == 0
    np.testing.assert_allclose(mass.sum(axis=1), 1.0)

    order = [3, 4, 2, 0, 1]
    reordered = helpers.compute_states(
        ds=prepared.isel(cells=order),
        target_root=str(tmp_path),
        chunks={"time": 1},
        save=False,
    )
    np.testing.assert_allclose(reordered["states"].fillna(0), mass[:, order])


def test_mixed_predictor_uses_adjoint_and_viterbi_follows_graph(emission, tmp_path):
    helpers.create_parameters(
        sigma=[8000 / 6371007.18],
        class_name="GaussianMixedHealpixAnalyse",
        n_steps=3,
        save_parameters=True,
        target_root=str(tmp_path),
    )
    estimator, prepared, _ = helpers._build_estimator(emission, str(tmp_path), None)
    predictor = estimator._get_predictors()[0]
    x, y = np.arange(5, dtype=float), np.array([1.0, 2.0, 2.5, 2.0, 0.0])
    np.testing.assert_allclose(
        np.dot(predictor.predict(x), y), np.dot(x, predictor.predict_backward(y))
    )
    assert not np.allclose(predictor.predict(y), predictor.predict_backward(y))

    path = estimator.decode(prepared, None, mode="viterbi_healpix", progress=False)
    assert path.df.geometry.iloc[0].x == pytest.approx(prepared["longitude"].values[0])
    assert path.df.geometry.iloc[-1].x == pytest.approx(prepared["longitude"].values[2])
    tracks = helpers.decode_trajectories(
        ds=emission,
        target_root=str(tmp_path),
        chunks={"time": 1},
        save=False,
        track_modes=["viterbi_healpix"],
    )
    assert len(tracks.trajectories) == 1


def test_mixed_contacts_reject_missing_cell(emission):
    emission["water_contact_from"] = ("water_contacts", [0, 1])
    prepared = helpers._prepare_movement_grid(emission, mixed=True)
    with pytest.raises(ValueError, match="absent from this grid"):
        helpers._get_predictor_factory(
            prepared,
            truncate=None,
            dims=["cells"],
            conv_method="HealpixAnalyseMixedConv",
        )
