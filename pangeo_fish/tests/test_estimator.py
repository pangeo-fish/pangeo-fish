import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xdggs
from tlz.functoolz import curry

from pangeo_fish.hmm.estimator import EagerEstimator
from pangeo_fish.hmm.optimize import EagerBoundsSearch
from pangeo_fish.hmm.prediction import Gaussian1DHealpix


@pytest.fixture
def sample_dataset():
    num_cells = 20  # Nombre de cellules
    num_time_steps = 10  # Nombre de pas de temps

    start_time = pd.Timestamp("2022-06-13T12:00:00")
    end_time = pd.Timestamp("2022-06-24T05:00:00")
    time = pd.date_range(start=start_time, end=end_time, periods=num_time_steps)

    # Paramètre Healpix
    nside = 2
    cell_ids = np.arange(4 * nside**2, 6 * nside**2)
    num_cells = cell_ids.size

    # Conversion en identifiants Healpix

    # Création de la variable initial (tout 0 sauf le centre)
    initial = np.zeros(num_cells)
    initial[len(initial) // 2] = 0.99  # Ajouter 0.99 au centre
    final = np.zeros(num_cells)
    final[(len(final) // 2) + 1] = 0.75
    mask = np.ones(num_cells)
    # Générer pdf(time, cells) avec somme = 1 pour chaque time step
    pdf = np.random.rand(num_time_steps, num_cells)  # Valeurs aléatoires
    pdf /= pdf.sum(axis=1, keepdims=True)  # Normalisation pour somme = 1

    ds = xr.Dataset(
        coords={
            "cell_ids": ("cells", cell_ids),
            "time": ("time", time),  # Ajout de la coordonnée time
        },
        data_vars={
            "pdf": (("time", "cells"), pdf),
            "initial": ("cells", initial),  # Variable initial
            "final": ("cells", final),
            "mask": ("cells", mask),
        },
    )
    ds.attrs["max_sigma"] = 0.014006609811820924
    return ds.dggs.decode(
        {"grid_name": "healpix", "level": 1, "indexing_scheme": "nested"}
    ).dggs.assign_latlon_coords()


@pytest.fixture
def sample_emission():
    sample_emission = xr.open_dataset(
        "test_unit_healpix_1D.zarr",
        engine="zarr",
        chunks={},
        inline_array=True,
        storage_options=None,
    )

    sample_emission["cell_ids"].attrs["grid_name"] = "healpix"
    sample_emission["cell_ids"].attrs["level"] = 12
    sample_emission = sample_emission.pipe(xdggs.decode)

    # Return the refined dataset
    return sample_emission.compute()


@pytest.fixture
def predictor_factory(sample_dataset):
    """
    Returns a predictor factory configured.
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


@pytest.mark.parametrize("sigma", [0.0004, 0.0002])
def test_eager_estimator_score(sample_dataset, predictor_factory, sigma):
    """
    Test the `score` method of EagerEstimator for different sigma values.
    """
    estimator = EagerEstimator(predictor_factory=predictor_factory, sigma=None)

    score = estimator.set_params(sigma=sigma).score(sample_dataset)

    assert isinstance(score, float), "Score should be a float."
    assert (
        score >= sample_dataset.sizes["time"]
    ), "Score should be equal to or bigger than the number of timesteps."


@pytest.mark.parametrize("sigma", [0.000001, 0.01, 0.005, 0.02, 0.03])
def test_eager_estimator_predict_proba(sample_dataset, predictor_factory, sigma):
    """
    Test the `predict_proba` method to ensure state probabilities are computed correctly.

    """
    estimator = EagerEstimator(predictor_factory=predictor_factory, sigma=sigma)
    state_probabilities = estimator.predict_proba(sample_dataset)
    print(state_probabilities[0])
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
def test_fit_sigma(sample_dataset, predictor_factory, tolerance):
    """
    Test to find best sigma with fit function.
    """
    estimator = EagerEstimator(sigma=None, predictor_factory=predictor_factory)
    optimizer = EagerBoundsSearch(
        estimator,
        (1e-4, 0.0004905038455501491),
        optimizer_kwargs={"disp": 3, "xtol": tolerance},
    )
    optimized = optimizer.fit(sample_dataset)
    # Print the type of optimized to debug
    print("Type of optimized:", type(optimized))

    if hasattr(optimized, "sigma"):
        result = optimized.sigma

        if isinstance(result, np.ndarray | float | int):
            assert not np.isnan(result).any(), "Optimized sigma is nan"
        else:
            raise ValueError("Optimized result is not in the expected format ")
