from dataclasses import dataclass, field
from typing import Any

import dask.array as da
import healpix_convolution as hc
import healpix_convolution.convolution
import numpy as np
import scipy.ndimage
from xarray.namedarray._typing import _arrayfunction_or_api as _ArrayLike
from xdggs.grid import DGGSInfo


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


@dataclass
class Predictor:
    def predict(self, X, *, mask=None):
        pass


@dataclass
class Gaussian2DCartesian(Predictor):
    sigma: float
    truncate: float = 4.0
    filter_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"mode": "constant", "cval": 0}
    )

    def predict(self, X, *, mask=None):
        filtered = gaussian_filter(X, sigma=self.sigma, **self.filter_kwargs)

        if mask is None:
            return filtered

        return np.where(mask, filtered)


@dataclass
class Gaussian1DHealpix(Predictor):
    cell_ids: _ArrayLike
    grid_info: DGGSInfo

    sigma: float
    truncate: float = 4.0
    kernel_size: int | None = None
    weights_threshold: float | None = None

    pad_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"mode": "constant", "constant_values": 0}
    )

    def __post_init__(self):
        ring = hc.kernels.gaussian.compute_ring(
            self.grid_info.resolution, self.sigma, self.truncate, self.kernel_size
        )
        self.padder = hc.padding.pad(
            self.cell_ids, self.grid_info, ring=ring, **self.pad_kwargs
        )
        self.kernel = hc.kernels.gaussian_kernel(
            self.cell_ids,
            resolution=self.grid_info.resolution,
            indexing_scheme=self.grid_info.indexing_scheme,
            sigma=self.sigma,
            truncate=self.truncate,
            kernel_size=self.kernel_size,
            weights_threshold=self.weights_threshold,
        )

    def predict(self, X, *, mask=None):
        padded = self.padder.apply(X)
        return hc.convolution.convolve(padded, self.kernel)
