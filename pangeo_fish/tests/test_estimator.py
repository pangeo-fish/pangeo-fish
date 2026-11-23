import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xdggs  # noqa: F401
from tlz.functoolz import curry

from pangeo_fish.hmm.estimator import EagerEstimator
from pangeo_fish.hmm.optimize import EagerBoundsSearch
from pangeo_fish.hmm.prediction import Gaussian1DHealpix


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


@pytest.fixture
def predictor_factory(sample_dataset):
    """
    Return a configured Gaussian1DHealpix predictor factory using sample dataset parameters.
    """
    return curry(
        Gaussian1DHealpix,
        cell_ids=sample_dataset["cell_ids"].data,
        grid_info=sample_dataset.dggs.grid_info,
        truncate=4.0,
        weights_threshold=1e-8,
        pad_kwargs={"mode": "constant", "constant_value": 0},
        optimize_convolution=True,
    )


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


def test_foscat_parameters_file_gives_clear_error(tmp_path):
    """
    A parameters file written with the removed Foscat1DHealpix predictor must fail
    with a message naming the replacement.
    """
    import json

    from pangeo_fish.helpers import _build_estimator

    params = {
        "sigma": [0.0004],
        "predictor_factory": {"class": "Foscat1DHealpix", "kwargs": {}},
    }
    (tmp_path / "parameters.json").write_text(json.dumps(params))

    with pytest.raises(ValueError, match="FFTGaussian1DHealpix"):
        _build_estimator(xr.Dataset(), str(tmp_path), None)


def test_old_updown_class_is_not_read_as_the_new_one(tmp_path):
    """
    "OldUpDownGaussian1DHealpix" contains the name of the new class: it must not be
    built as "UpDownGaussian1DHealpix", and `create_parameters` must not write it.
    """
    import json

    from pangeo_fish.helpers import _build_estimator, create_parameters

    params = {
        "sigma": [0.0004],
        "predictor_factory": {"class": "OldUpDownGaussian1DHealpix", "kwargs": {}},
    }
    (tmp_path / "parameters.json").write_text(json.dumps(params))

    with pytest.raises(ValueError, match="comparison only"):
        _build_estimator(xr.Dataset(), str(tmp_path), None)
    with pytest.raises(ValueError, match="comparison only"):
        create_parameters([0.0004], class_name="OldUpDownGaussian1DHealpix")


def test_removed_predictor_reported_before_converting_the_emission(
    tmp_path, sample_dataset
):
    """The removal message comes first, also for an emission with a cells dim."""
    import json

    from pangeo_fish.helpers import _build_estimator

    params = {"sigma": [0.0004], "predictor_factory": {"class": "Foscat1DHealpix"}}
    (tmp_path / "parameters.json").write_text(json.dumps(params))
    with pytest.raises(ValueError, match="FFTGaussian1DHealpix"):
        _build_estimator(sample_dataset, str(tmp_path), None)


def test_updown_predictor_without_ellipsoid_uses_a_sphere(sample_dataset):
    """Numpy cell ids and a grid without an ellipsoid: treated as a sphere."""
    from pangeo_fish.hmm.prediction import UpDownGaussian1DHealpix

    predictor = UpDownGaussian1DHealpix(
        cell_ids=sample_dataset["cell_ids"].data,
        grid_info=sample_dataset.dggs.grid_info,
        sigma=0.001,
    )
    assert predictor.ellipsoid == "sphere"
    state = sample_dataset["initial"].values
    np.testing.assert_allclose(predictor.predict(state).sum(), 1.0, rtol=1e-3)


@pytest.mark.parametrize("factory", [None, "missing"])
def test_parameters_file_without_predictor_factory(tmp_path, factory):
    import json

    from pangeo_fish.helpers import _build_estimator

    params = {"sigma": [0.0004]}
    if factory is None:
        params["predictor_factory"] = None
    (tmp_path / "parameters.json").write_text(json.dumps(params))
    with pytest.raises(ValueError, match="no predictor_factory"):
        _build_estimator(xr.Dataset(), str(tmp_path), None)
