from dataclasses import dataclass, field
from typing import Any

import dask.array as da
import numpy as np
import scipy.ndimage
import torch
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
        import healpix_convolution as hc
        import healpix_convolution.padding
        import opt_einsum

        ring = hc.kernels.gaussian.compute_ring(
            self.grid_info.level, self.sigma, self.truncate, self.kernel_size
        )
        self.padder = hc.padding.pad(
            self.cell_ids, grid_info=self.grid_info, ring=ring, **self.pad_kwargs
        )
        self.new_cell_ids, self.kernel = hc.kernels.gaussian_kernel(
            self.cell_ids,
            grid_info=self.grid_info,
            sigma=self.sigma,
            truncate=self.truncate,
            kernel_size=self.kernel_size,
            weights_threshold=self.weights_threshold,
        )

        if self.optimize_convolution:
            self.convolve = opt_einsum.contract_expression(
                "...a,ba->...b", self.padder.cell_ids.shape, self.kernel, constants=[1]
            )
        else:
            from healpix_convolution.convolution import convolve

            self.convolve = curry(convolve, kernel=self.kernel)

    def predict(self, X, *, mask=None):
        padded = self.padder.apply(X)
        filtered = self.convolve(padded)

        if mask is None:
            return filtered

        return np.where(mask, filtered, 0)


import warnings


@dataclass
class Foscat1DHealpix(Predictor):
    cell_ids: _ArrayLike
    grid_info: DGGSInfo

    sigma: float  # en radians
    kernel_size: int | None = None
    max_kernel_size: int | None = 33

    def __post_init__(self):
        import foscat.SphericalStencil as sc

        nside = 2**self.grid_info.level

        # # sigma_opt : conversion de sigma (radians) vers pixels
        sigma_opt = (
            self.sigma / np.sqrt(np.pi)
        ) * nside  # parfois ajouter un 3* à coté de pi 3*

        radius = int(
            np.ceil(3 * sigma_opt)
        )  # troncature à 3 sigma, indépendant du facteur retiré ci-dessus
        kernel_size = 2 * radius + 1

        if self.max_kernel_size is not None and kernel_size > self.max_kernel_size:
            warnings.warn(
                f"kernel_size tronqué de {kernel_size} à {self.max_kernel_size} "
                f"(sigma_opt={sigma_opt:.2f}) : le noyau gaussien réel sera tronqué, "
                f"le sigma effectif sera plus petit que demandé pour ce sigma.",
                RuntimeWarning,
            )
            kernel_size = self.max_kernel_size
            kernel_size -= 1 - kernel_size % 2

        self.kernel_size = kernel_size
        # self.kernel_size = 33#

        self.stencil = sc.SphericalStencil(
            nside, int(self.kernel_size), cell_ids=self.cell_ids
        )

        xx, yy = np.meshgrid(
            np.arange(self.kernel_size) - self.kernel_size // 2,
            np.arange(self.kernel_size) - self.kernel_size // 2,
        )
        W = np.exp(-(xx**2 + yy**2) / (sigma_opt**2))
        W = W / W.sum()

        self.W_tensor = self.stencil.to_tensor(W).reshape(1, 1, self.kernel_size**2)

    def _ensure_bcp(self, arr: np.ndarray):
        """
        Ensure (B, C, P).
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
        if mask is not None:
            filtered = np.where(mask, filtered, 0)
        return filtered


@dataclass
@dataclass
class UpDownGaussian1DHealpix(Predictor):
    cell_ids: _ArrayLike
    grid_info: DGGSInfo

    sigma: float  # en radians
    kernel_sz: int | None = None  # taille nominale du noyau (résolution fine)
    max_compact_kernel_sz: int = 7  # borne sur le noyau compact (résolution grossière)
    device: str = "cpu"
    dtype: any = torch.float32

    def __post_init__(self):
        import healpy as hp
        from healpix_analyse import LargeConv

        nside = 2**self.grid_info.level

        # --- sigma_opt : sigma (radians) -> pixels du grid fin ---
        sigma_opt = (self.sigma / np.sqrt(np.pi)) * nside  # 3*
        radius = int(np.ceil(3 * sigma_opt))
        if self.kernel_sz is None:
            self.kernel_sz = 2 * radius + 1

        # --- 1. Construction de l'opérateur multi-résolution ---
        self.layer = LargeConv(
            level=self.grid_info.level,
            in_channels=1,
            out_channels=1,
            kernel_sz=self.kernel_sz,
            max_compact_kernel_sz=self.max_compact_kernel_sz,
            cell_ids=self.cell_ids,
            ellipsoid="sphere",
            weight_norm="l1",
            up_norm="col_l1",
            device=self.device,
            dtype=self.dtype,
        )

        coarse_nside = self.layer.coarse_nside
        k = self.layer.compact_kernel_sz
        n_cells = len(self.cell_ids)
        centre_index = n_cells // 2

        # --- 2. Mesure du flou introduit par Down/Up seuls (noyau compact = impulsion identité) ---
        identity_kernel = np.zeros((1, 1, k * k), dtype=np.float32)
        identity_kernel[0, 0, k * k // 2] = 1.0
        self.layer.set_compact_kernel(
            identity_kernel, bias=np.zeros(1, dtype=np.float32), requires_grad=False
        )

        impulse = np.zeros(n_cells, dtype=np.float32)
        impulse[centre_index] = 1.0
        response_downup_only = self.layer(impulse)

        lon, lat = hp.pix2ang(nside, self.cell_ids, nest=True, lonlat=True)  # degrés
        sigma_downup_deg = self._measure_effective_sigma(
            response_downup_only, lon, lat, centre_index
        )
        sigma_downup_rad = np.radians(sigma_downup_deg)

        # --- 3. Résidu en quadrature (tout en radians) ---
        residual2 = self.sigma**2 - sigma_downup_rad**2
        if residual2 <= 0:
            # Down/Up floute déjà plus que ce qui est demandé : noyau compact quasi ponctuel
            warnings.warn(
                f"sigma demandé ({self.sigma:.2e} rad) est inférieur ou égal au flou "
                f"intrinsèque du Down/Up ({sigma_downup_rad:.2e} rad) pour cette géométrie. "
                f"Le noyau compact sera quasi ponctuel, le flou effectif sera dominé par "
                f"le Down/Up et ne pourra pas être réduit davantage.",
                RuntimeWarning,
            )
            sigma_residual_rad = 0.0
        else:
            sigma_residual_rad = np.sqrt(residual2)

        # --- 4. Conversion du résidu (radians) -> pixels du grid grossier ---
        sigma_compact = (sigma_residual_rad / np.sqrt(np.pi)) * coarse_nside
        sigma_compact = max(
            sigma_compact, 1e-6
        )  # éviter une division par 0 dans le noyau

        # --- 5. Construction du vrai noyau gaussien compact ---
        xx, yy = np.meshgrid(
            np.arange(k) - k // 2,
            np.arange(k) - k // 2,
        )
        W = np.exp(-(xx**2 + yy**2) / (sigma_compact**2))
        W = W / W.sum()
        W = W.reshape(1, 1, k**2).astype(np.float32)

        self.layer.set_compact_kernel(
            W, bias=np.zeros(1, dtype=np.float32), requires_grad=False
        )

        # gardés pour diagnostic / debug après coup
        self.sigma_downup_rad = sigma_downup_rad
        self.sigma_compact_px = sigma_compact

    @staticmethod
    def _measure_effective_sigma(response, lon, lat, centre_index):
        # lon, lat en degrés (lonlat=True)
        c_lon, c_lat = lon[centre_index], lat[centre_index]
        dlon = ((lon - c_lon + 180.0) % 360.0) - 180.0
        dlat = lat - c_lat
        r2 = dlon**2 + dlat**2
        w = np.abs(response)
        w = w / w.sum()
        sigma2 = np.sum(w * r2) / 2.0
        return np.sqrt(sigma2)

    def _ensure_bcp(self, arr):
        n_cells = self.cell_ids.shape[0]
        a = np.array(arr)
        if a.ndim == 1:
            return a.reshape(1, 1, -1), ("1d", a.shape)
        if a.ndim == 2:
            if a.shape[1] == n_cells:
                return a.reshape(a.shape[0], 1, a.shape[1]), ("2d_bp", a.shape)
            if a.shape[0] == n_cells:
                return a.reshape(1, 1, a.shape[0]), ("2d_p?", a.shape)
            return a.reshape(1, 1, -1), ("2d_fallback", a.shape)
        if a.ndim == 3:
            return a, ("3d", a.shape)
        raise ValueError(f"Unsupported input ndim={a.ndim} for convolution")

    def _restore_shape(self, out, original_info):
        kind, orig_shape = original_info
        B, C, P = out.shape
        if kind == "1d":
            return out.reshape(
                P,
            )
        if kind == "2d_bp":
            return out.reshape(orig_shape[0], orig_shape[1])
        if kind == "2d_fallback":
            return out.reshape(1, P)
        return out

    def predict(self, X, mask=None):
        bcp, original_info = self._ensure_bcp(X)
        out = self.layer(bcp)
        out_np = out if isinstance(out, np.ndarray) else out.detach().cpu().numpy()

        filtered = self._restore_shape(out_np, original_info)
        if mask is not None:
            filtered = np.where(mask, filtered, 0)
        return filtered
