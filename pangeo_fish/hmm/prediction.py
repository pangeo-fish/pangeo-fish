from dataclasses import dataclass, field
from typing import Any

import dask.array as da
import numpy as np
import scipy.ndimage
from tlz.functoolz import curry
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

        return np.where(mask, filtered, 0)


from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Gaussian1DHealpix(Predictor):
    cell_ids: _ArrayLike
    grid_info: DGGSInfo

    sigma: float
    truncate: float = 4.0
    kernel_size: int | None = None
    weights_threshold: float | None = None

    pad_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"mode": "constant", "constant_value": 0}
    )
    optimize_convolution: bool = True

    def __post_init__(self):
        import foscat.SphericalStencil as sc

        self.kernel_size = 33
        nside = 2**self.grid_info.level

        self.stencil = sc.SphericalStencil(
            nside, int(self.kernel_size), cell_ids=self.cell_ids
        )

        sigma_opt = (self.sigma / np.sqrt(np.pi)) * nside
        xx, yy = np.meshgrid(
            np.arange(self.kernel_size) - self.kernel_size // 2,
            np.arange(self.kernel_size) - self.kernel_size // 2,
        )
        W = np.exp(-(xx**2 + yy**2) / (sigma_opt**2))
        W = W / W.sum()
        print("sigma", self.sigma)
        print("sigma_opt", sigma_opt)
        print(W)
        print("kernel cut:", W[:, self.kernel_size // 2])
        self.W_tensor = self.stencil.to_tensor(W).reshape(1, 1, self.kernel_size**2)

        class _DummyPadder:
            def __init__(self, cell_ids):
                self.cell_ids = cell_ids

            def apply(self, X):
                return np.array(X)

        self.padder = _DummyPadder(self.cell_ids)

    def _ensure_bcp(self, arr: np.ndarray):
        """
        Garantit que arr est en (B, C, P).
        """
        n_cells = self.cell_ids.shape[0]
        a = np.array(arr)
        if a.ndim == 1:
            return a.reshape(1, 1, -1), ("1d", a.shape)
        if a.ndim == 2:
            # cas (B, P)
            if a.shape[1] == n_cells:
                return a.reshape(a.shape[0], 1, a.shape[1]), ("2d_bp", a.shape)
            # cas (P, something) improbable : si first dim correspond à n_cells -> (1,1,P)
            if a.shape[0] == n_cells:
                return a.reshape(1, 1, a.shape[0]), ("2d_p?", a.shape)
            # fallback
            return a.reshape(1, 1, -1), ("2d_fallback", a.shape)
        if a.ndim == 3:
            return a, ("3d", a.shape)
        raise ValueError(f"Unsupported input ndim={a.ndim} for convolution")

    def _restore_shape(self, out: np.ndarray, original_info):
        """
        `out` (of shape (B,C,P)) to 1D
        """
        kind, orig_shape = original_info
        B, C, P = out.shape
        if kind == "1d":
            # renvoyer (P,)
            return out.reshape(
                P,
            )
        if kind == "2d_bp":
            # origine (B, P)
            return out.reshape(orig_shape[0], orig_shape[1])
        if kind == "2d_fallback":
            return out.reshape(1, P)
        # 3d : conserver
        return out

    def predict(self, X, mask=None):

        bcp, original_info = self._ensure_bcp(X)
        im_t = self.stencil.to_tensor(bcp)

        out_t = self.stencil.Convol_torch(im_t, self.W_tensor)
        out_np = self.stencil.to_numpy(out_t)

        filtered = self._restore_shape(out_np, original_info)
        return filtered
