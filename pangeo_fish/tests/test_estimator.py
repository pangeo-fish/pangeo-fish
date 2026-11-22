import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xdggs  # noqa: F401
from tlz.functoolz import curry

from pangeo_fish.hmm.estimator import EagerEstimator
from pangeo_fish.hmm.optimize import EagerBoundsSearch
from pangeo_fish.hmm.prediction import UpDownGaussian1DHealpix


@pytest.fixture
def sample_dataset():
    """
    Create a sample xarray dataset with random probabilities, initial/final states, and a mask.
    """
    num_time_steps = 10
    start_time = pd.Timestamp("2022-06-13T12:00:00")
    end_time = pd.Timestamp("2022-06-24T05:00:00")
    time = pd.date_range(start=start_time, end=end_time, periods=num_time_steps)

    level = 2
    cell_ids = np.arange(4 * 4**level, 6 * 4**level)

    initial = np.zeros_like(cell_ids, dtype="float64")
    initial[len(initial) // 2] = 0.80
    initial[len(initial) // 2 + 1] = 0.20
    final = np.zeros_like(cell_ids, dtype="float64")
    final[(len(final) // 2)] = 0.75

    mask = np.full_like(cell_ids, fill_value=True, dtype=bool)

    rng = np.random.default_rng(seed=0)
    pdf = rng.random(size=(num_time_steps, cell_ids.size))
    pdf /= pdf.sum(axis=1, keepdims=True)

    ds = xr.Dataset(
        coords={"cell_ids": ("cells", cell_ids), "time": ("time", time)},
        data_vars={
            "pdf": (("time", "cells"), pdf),
            "initial": ("cells", initial),
            "final": ("cells", final),
            "mask": ("cells", mask),
            "predictor_index": ("time", np.zeros(num_time_steps, dtype=np.int32)),
        },
    )
    return ds.dggs.decode(
        {"grid_name": "healpix", "level": level, "indexing_scheme": "nested"}
    ).dggs.assign_latlon_coords()


def make_predictor_factory(ds):
    """UpDownGaussian1DHealpix predictor factory for ``ds``."""
    return curry(
        UpDownGaussian1DHealpix,
        cell_ids=ds["cell_ids"].data,
        grid_info=ds.dggs.grid_info,
        device="cpu",
        ellipsoid="sphere",
    )


@pytest.fixture
def predictor_factory(sample_dataset):
    return make_predictor_factory(sample_dataset)


@pytest.mark.parametrize("sigma", [[0.0004], [0.0002]])
def test_eager_estimator_score(sample_dataset, predictor_factory, sigma):
    """
    Test that the score method returns a float and meets expected criteria for given sigma values.
    """
    estimator = EagerEstimator(predictor_factory=predictor_factory, sigma=None)
    score = estimator.set_params(sigma=sigma).score(sample_dataset)
    assert isinstance(score, float), "Score should be a float."
    assert (
        score >= sample_dataset.sizes["time"]
    ), "Score should be equal to or bigger than the number of timesteps."


@pytest.mark.parametrize("sigma", [[0.0001], [0.01], [0.005], [0.02]])
def test_eager_estimator_predict_proba(sample_dataset, predictor_factory, sigma):
    """
    Test that `predict_proba` computes positive state probabilities summing to 1 for each time step.
    """
    estimator = EagerEstimator(predictor_factory=predictor_factory, sigma=sigma)
    state_probabilities = estimator.predict_proba(sample_dataset)
    print(state_probabilities[9])
    assert (
        state_probabilities > 0
    ).sum() > 0, "Probabilities should contain valid positive values."
    assert np.allclose(
        state_probabilities.sum(axis=1), 1
    ), "The sum for each time step should be 1."


@pytest.mark.parametrize("bounds", [(1e-4, 1), (1, 5), (0.1, 10)])
@pytest.mark.parametrize("tolerance", [1e-3, 1e-6])
def test_eager_bounds_search(sample_dataset, predictor_factory, bounds, tolerance):
    """
    Test that the EagerBoundsSearch optimizer instance is created correctly with given bounds and tolerance.
    """
    estimator = EagerEstimator(sigma=None, predictor_factory=predictor_factory)
    optimizer = EagerBoundsSearch(
        estimator, bounds, optimizer_kwargs={"disp": 3, "xtol": tolerance}
    )
    assert isinstance(optimizer, EagerBoundsSearch)


@pytest.mark.parametrize("tolerance", [1e-3, 1e-4])
def test_fit_sigma(sample_dataset, predictor_factory, tolerance):
    """
    Test that the `fit` method of EagerBoundsSearch finds a valid sigma without NaN values.
    """
    estimator = EagerEstimator(sigma=None, predictor_factory=predictor_factory)
    optimizer = EagerBoundsSearch(
        estimator,
        (1e-4, 0.0004905038455501491),
        optimizer_kwargs={"disp": 3, "xtol": tolerance},
    )
    optimized = optimizer.fit_single_parameter(sample_dataset)
    print("Type of optimized:", type(optimized))
    if hasattr(optimized, "sigma"):
        result = optimized.sigma
    if isinstance(result, (np.ndarray, float, int, list)):
        arr = np.atleast_1d(result)
        assert not np.isnan(arr).any(), "Optimized sigma is nan"
    else:
        raise ValueError("Optimized result is not in the expected format ")


def test_updown_predictor_without_ellipsoid(sample_dataset):
    """A grid without an ellipsoid is treated as a sphere (the default path)."""
    from pangeo_fish.helpers import _get_predictor_factory

    factory = _get_predictor_factory(
        sample_dataset, None, ["cells"], "LargeHealpixConv"
    )
    predictor = factory(sigma=0.001)
    assert predictor.ellipsoid == "sphere"
    state = sample_dataset["initial"].values
    np.testing.assert_allclose(predictor.predict(state).sum(), 1.0, rtol=1e-3)


@pytest.mark.parametrize("where", ["create_parameters", "conv_method", "json"])
def test_foscat_is_removed(tmp_path, sample_dataset, where):
    import json

    from pangeo_fish import helpers

    with pytest.raises(ValueError, match="UpDownGaussian1DHealpix"):
        if where == "create_parameters":
            helpers.create_parameters(sigma=[0.001], class_name="Foscat1DHealpix")
        elif where == "conv_method":
            helpers._get_predictor_factory(
                sample_dataset, None, ["cells"], "FoscatConv"
            )
        else:
            params = {
                "sigma": [0.001],
                "sigma_indices": [list(range(sample_dataset.sizes["time"]))],
                "predictor_factory": {"class": "Foscat1DHealpix", "kwargs": {}},
            }
            (tmp_path / "parameters.json").write_text(json.dumps(params))
            helpers._build_estimator(sample_dataset, str(tmp_path), None)
