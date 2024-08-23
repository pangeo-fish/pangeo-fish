from dataclasses import dataclass
from typing import Any

import dask.array as da
import numpy as np
import scipy.ndimage


def gaussian_filter(X, sigma, **kwargs):
    if isinstance(X, da.Array) and X.npartitions > 1:
        import dask_image.ndfilters

        return dask_image.ndfilters.gaussian_filter(X, sigma=sigma, **kwargs)
    elif isinstance(X, da.Array):
        return X.map_blocks(
            scipy.ndimage.gaussian_filter,
            sigma=sigma,
            meta=np.array((), dtype=X.dtype),
            **kwargs,
        )
    else:
        return scipy.ndimage.gaussian_filter(X, sigma=sigma, **kwargs)


@dataclass(frozen=True)
class Predictor:
    def predict(self, X, *, mask=None):
        pass


@dataclass
class Gaussian2DCartesian(Predictor):
    sigma: float
    truncate: float = 4.0
    filter_kwargs: dict[str, Any] = {"mode": "constant", "cval": 0}

    def predict(self, X, *, mask=None):
        filtered = gaussian_filter(X, sigma=self.sigma, **self.filter_kwargs)

        if mask is None:
            return filtered

        return np.where(mask, filtered)
