import numpy as np
import pytest
import xarray as xr
import xdggs
from tlz.functoolz import curry

from pangeo_fish.hmm.estimator import EagerEstimator
from pangeo_fish.hmm.optimize import EagerBoundsSearch
from pangeo_fish.hmm.prediction import Gaussian1DHealpix

pytestmark = pytest.mark.skip(reason="Skipping all tests in this file temporarily.")


@pytest.fixture
def sample_emission():
    sample_emission = xr.open_dataset(
        "test_unit_healpix_1D.zarr",
        engine="zarr",
        chunks={},
        inline_array=True,
        storage_options=None,
    )
    # Set required attributes cleanly
    sample_emission["cell_ids"].attrs["grid_name"] = "healpix"
    # Set required attributes cleanly
    sample_emission["cell_ids"].attrs["level"] = 12
    sample_emission = sample_emission.pipe(xdggs.decode)

    # Return the refined dataset
    return sample_emission.compute()


@pytest.fixture
def predictor_factory(sample_emission):
    """
    Returns a predictor factory configured.
    """
    return curry(
        Gaussian1DHealpix,
        cell_ids=sample_emission["cell_ids"].data,
        grid_info=sample_emission.dggs.grid_info,
        truncate=4.0,
        weights_threshold=1e-8,
        pad_kwargs={"mode": "constant", "constant_value": 0},
        optimize_convolution=True,
    )


@pytest.mark.parametrize("sigma", [0.0004, 0.0002])
def test_eager_estimator_score(sample_emission, predictor_factory, sigma):
    """
    Test the `score` method of EagerEstimator for different sigma values.
    """
    estimator = EagerEstimator(predictor_factory=predictor_factory, sigma=None)

    score = estimator.set_params(sigma=sigma).score(sample_emission)

    assert isinstance(score, float), "Score should be a float."
    assert score >= 0, "Score should not be negative."


@pytest.mark.parametrize("sigma", [0.0004, 0.0002])
def test_eager_estimator_predict_proba(sample_emission, predictor_factory, sigma):
    """
    Test the `predict_proba` method to ensure state probabilities are computed correctly.

    """
    estimator = EagerEstimator(predictor_factory=predictor_factory, sigma=sigma)
    state_probabilities = estimator.predict_proba(sample_emission)

    assert (
        state_probabilities > 0
    ).sum() > 0, "Probabilities should contain valid positive values."


@pytest.mark.parametrize("bounds", [(1e-4, 1), (1, 5), (0.1, 10)])
@pytest.mark.parametrize("tolerance", [1e-3, 1e-6])
def test_eager_bounds_search(sample_emission, predictor_factory, bounds, tolerance):
    """
    Test the EagerBoundsSearch optimizer for finding the optimal sigma.
    """
    estimator = EagerEstimator(sigma=None, predictor_factory=predictor_factory)
    optimizer = EagerBoundsSearch(
        estimator, bounds, optimizer_kwargs={"disp": 3, "xtol": tolerance}
    )

    assert isinstance(optimizer, EagerBoundsSearch)


@pytest.mark.parametrize("tolerance", [1e-3, 1e-4])
def test_fit_sigma(sample_emission, predictor_factory, tolerance):
    """
    Test to find best sigma with fit function.
    """
    estimator = EagerEstimator(sigma=None, predictor_factory=predictor_factory)
    optimizer = EagerBoundsSearch(
        estimator,
        (1e-4, 0.0004905038455501491),
        optimizer_kwargs={"disp": 3, "xtol": tolerance},
    )
    optimized = optimizer.fit(sample_emission)
    # Print the type of optimized to debug
    print("Type of optimized:", type(optimized))

    if hasattr(optimized, "sigma"):
        result = optimized.sigma

        if isinstance(result, np.ndarray | float | int):
            assert not np.isnan(result).any(), "Optimized sigma is nan"
        else:
            raise ValueError("Optimized result is not in the expected format ")
