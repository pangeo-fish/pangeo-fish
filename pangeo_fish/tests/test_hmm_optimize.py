from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from pangeo_fish.hmm import optimize
from pangeo_fish.hmm.estimator import EagerEstimator


@dataclass
class Predictor:
    sigma: float
    diff: np.ndarray

    def predict(self, X):
        return X + 1 * self.diff


class TestEagerBoundsSearch:
    @pytest.mark.parametrize(
        "bounds",
        (
            (0, 1),
            (1e-5, 5),
            (1e-5, xr.DataArray(7.3)),
        ),
    )
    def test_init(self, bounds):
        estimator = EagerEstimator(sigma=None, predictor_factory=Predictor)
        optimizer = optimize.EagerBoundsSearch(estimator, bounds)

        assert isinstance(optimizer, optimize.EagerBoundsSearch)
        assert (
            isinstance(optimizer.param_bounds, tuple)
            and len(optimizer.param_bounds) == 2
        )
        assert all(
            isinstance(v, (int, float, np.number)) for v in optimizer.param_bounds
        )
