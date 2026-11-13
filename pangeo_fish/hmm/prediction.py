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

        # # sigma_opt: convert sigma (radians) to pixels
        sigma_opt = (
            self.sigma / np.sqrt(np.pi)
        ) * nside  # sometimes add a 3* next to pi

        radius = int(
            np.ceil(3 * sigma_opt)
        )  # truncation at 3 sigma, independent of the factor removed above
        kernel_size = 2 * radius + 1

        if self.max_kernel_size is not None and kernel_size > self.max_kernel_size:
            warnings.warn(
                f"kernel_size truncated from {kernel_size} to {self.max_kernel_size} "
                f"(sigma_opt={sigma_opt:.2f}): the actual Gaussian kernel will be truncated, "
                f"so the effective sigma will be smaller than requested for this sigma.",
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
class OldUpDownGaussian1DHealpix(Predictor):
    cell_ids: _ArrayLike
    grid_info: DGGSInfo

    sigma: float  # in radians
    kernel_sz: int | None = None  # nominal kernel size (fine resolution)
    max_compact_kernel_sz: int = (
        7  # upper bound on the compact kernel (coarse resolution)
    )
    ellipsoid: str = None
    device: str = "cpu"
    dtype: any = torch.float32

    def __post_init__(self):
        import healpix_geo.nested as hpg_n
        from healpix_analyse import LargeConv

        nside = 2**self.grid_info.level

        # --- sigma_opt: sigma (radians) -> fine grid pixels ---
        sigma_opt = (self.sigma / np.sqrt(np.pi)) * nside  # 3*
        radius = int(np.ceil(3 * sigma_opt))
        if self.kernel_sz is None:
            self.kernel_sz = 2 * radius + 1
        if self.ellipsoid is None:
            self.ellipsoid = self.cell_ids.attrs.get("ellipsoid")
        # --- 1. Build the multi-resolution operator ---
        self.layer = LargeConv(
            level=self.grid_info.level,
            in_channels=1,
            out_channels=1,
            kernel_sz=self.kernel_sz,
            max_compact_kernel_sz=self.max_compact_kernel_sz,
            cell_ids=self.cell_ids,
            ellipsoid=self.ellipsoid,
            weight_norm="l1",
            up_norm="col_l1",
            device=self.device,
            dtype=self.dtype,
        )

        coarse_nside = self.layer.coarse_nside
        k = self.layer.compact_kernel_sz
        n_cells = len(self.cell_ids)
        centre_index = n_cells // 2

        # --- 2. Measure the blur introduced by Down/Up alone (compact kernel = identity impulse) ---
        identity_kernel = np.zeros((1, 1, k * k), dtype=np.float32)
        identity_kernel[0, 0, k * k // 2] = 1.0
        self.layer.set_compact_kernel(
            identity_kernel, bias=np.zeros(1, dtype=np.float32), requires_grad=False
        )

        impulse = np.zeros(n_cells, dtype=np.float32)
        impulse[centre_index] = 1.0
        response_downup_only = self.layer(impulse)

        # lon, lat = hp.pix2ang(nside, self.cell_ids, nest=True, lonlat=True)  # degrees
        lon, lat = hpg_n.healpix_to_lonlat(
            self.cell_ids, self.grid_info.level, ellipsoid=self.layer.ellipsoid
        )

        sigma_downup_deg = self._measure_effective_sigma(
            response_downup_only, lon, lat, centre_index
        )
        sigma_downup_rad = np.radians(sigma_downup_deg)

        # --- 3. Residual in quadrature (everything in radians) ---
        residual2 = self.sigma**2 - sigma_downup_rad**2
        if residual2 <= 0:
            # Down/Up already blurs more than requested: near-pointwise compact kernel
            warnings.warn(
                f"Requested sigma ({self.sigma:.2e} rad) is less than or equal to the "
                f"intrinsic blur of Down/Up ({sigma_downup_rad:.2e} rad) for this geometry. "
                f"The compact kernel will be nearly pointwise; the effective blur will be "
                f"dominated by Down/Up and cannot be reduced further.",
                RuntimeWarning,
            )
            sigma_residual_rad = 0.0
        else:
            sigma_residual_rad = np.sqrt(residual2)

        # --- 4. Convert the residual (radians) -> coarse grid pixels ---
        sigma_compact = (sigma_residual_rad / np.sqrt(np.pi)) * coarse_nside
        sigma_compact = max(sigma_compact, 1e-6)  # avoid division by 0 in the kernel

        # --- 5. Build the actual compact Gaussian kernel ---
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

        # kept for diagnostics / debugging afterwards
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
    def check_impulse_response(self, centre_index=None):
        """Diagnostic: apply the filter to a Dirac and measure the effective sigma.

        Returns a dict with the measured sigma (radians), the ratio to the
        requested sigma, and the total mass of the response.
        """
        import healpix_geo.nested as hpg_n

        n = len(self.cell_ids)
        c = n // 2 if centre_index is None else int(centre_index)

        x = np.zeros(n, dtype=np.float32)
        x[c] = 1.0
        resp = np.asarray(self.predict(x)).ravel()

        lon, lat = hpg_n.healpix_to_lonlat(
            np.asarray(self.cell_ids), self.grid_info.level, ellipsoid=self.layer.ellipsoid
        )

        # drop float32 noise so it does not count as weight at large distances
        resp_clean = np.clip(resp, 0.0, None)
        resp_clean[resp_clean < 1e-6 * resp_clean.max()] = 0.0

        s_measured = self._effective_sigma(resp_clean, lon, lat, c)
        return {
            "sigma_requested": self.sigma,
            "sigma_measured": s_measured,
            "ratio": s_measured / self.sigma,
            "mass": float(resp.sum()),
            "sigma_downup": self.sigma_downup_rad,
            "n_levels": self.layer.n_levels,
            "cap_used": self.max_compact_kernel_sz_used,
        }


import healpix_geo.nested as hpg_n
from healpix_analyse import LargeConv
@dataclass
class UpDownGaussian1DHealpix(Predictor):
    """Gaussian filter on a HEALPix patch, built on a multi-resolution LargeConv.

    The target blur ``sigma`` is split in quadrature between the intrinsic blur
    of the Down/Up operators and a compact Gaussian kernel applied at the
    coarse resolution.
    """

    cell_ids: _ArrayLike
    grid_info: DGGSInfo

    sigma: float  # target standard deviation, in radians
    kernel_sz: int | None = None  # nominal kernel size (fine resolution)
    max_compact_kernel_sz: int = 7  # starting cap on the compact kernel size
    max_compact_kernel_sz_cap: int = 15  # upper bound for the cap search
    ellipsoid: str | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    # Single place for the radians -> pixels conversion.
    # To be validated against the step actually used by HealPixConv.
    @staticmethod
    def _rad_to_px(x_rad, nside):
        return x_rad * nside / np.sqrt(np.pi / 3)

    def _build_layer(self, max_k):

        return LargeConv(
            level=self.grid_info.level,
            in_channels=1,
            out_channels=1,
            kernel_sz=self.kernel_sz,
            max_compact_kernel_sz=max_k,
            cell_ids=self.cell_ids,
            ellipsoid=self.ellipsoid,
            weight_norm="l1",
            up_norm="col_l1",
            device=self.device,
            dtype=self.dtype,
        )

    def _measure_downup(self, layer):
        """Measure the blur (radians) introduced by Down/Up alone."""
        import healpix_geo.nested as hpg_n

        k, n = layer.compact_kernel_sz, len(self.cell_ids)
        c = n // 2  # index of the (assumed) central pixel

        # Identity compact kernel: only Down/Up acts on the impulse
        ident = np.zeros((1, 1, k * k), np.float32)
        ident[0, 0, k * k // 2] = 1.0
        layer.set_compact_kernel(
            ident, bias=np.zeros(1, np.float32), requires_grad=False
        )

        impulse = np.zeros(n, np.float32)
        impulse[c] = 1.0
        resp = layer(impulse)
        resp = resp if isinstance(resp, np.ndarray) else resp.detach().cpu().numpy()

        lon, lat = hpg_n.healpix_to_lonlat(
            self.cell_ids, self.grid_info.level, ellipsoid=layer.ellipsoid
        )
        return self._effective_sigma(resp.ravel(), lon, lat, c)  # radians

    @staticmethod
    def _effective_sigma(resp, lon, lat, c):
        """Effective per-axis sigma (radians) of a response around pixel ``c``.

        lon, lat are in degrees. Uses the true angular distance (haversine-like)
        so that no latitude correction is needed.
        """
        lo, la = np.radians(lon), np.radians(lat)
        cosd = np.sin(la[c]) * np.sin(la) + np.cos(la[c]) * np.cos(la) * np.cos(
            lo - lo[c]
        )
        d2 = np.arccos(np.clip(cosd, -1, 1)) ** 2  # squared angular distance
        w = np.abs(resp)
        w = w / w.sum()
        return np.sqrt(np.sum(w * d2) / 2.0)  # per-axis sigma

    def __post_init__(self):
        nside = 2**self.grid_info.level

        if self.ellipsoid is None:
            self.ellipsoid = self.cell_ids.attrs.get("ellipsoid")

        # Nominal fine-resolution kernel size: radius = 3 * sigma (in pixels)
        if self.kernel_sz is None:
            sigma_px = self._rad_to_px(self.sigma, nside)
            self.kernel_sz = 2 * int(np.ceil(3 * sigma_px)) + 1

        # Increasing caps: keep the first one whose Down/Up blur is below sigma.
        # A larger cap means fewer levels, hence less parasitic blur.
        caps = sorted(
            {
                self.max_compact_kernel_sz,
                *range(
                    self.max_compact_kernel_sz,
                    self.max_compact_kernel_sz_cap + 1,
                    2,
                ),
            }
        )
        for cap in caps:
            layer = self._build_layer(cap)
            s_du = self._measure_downup(layer)
            if s_du < self.sigma:
                break
        else:
            warnings.warn(
                f"Requested sigma ({self.sigma:.2e} rad) is lower than or equal to "
                f"the minimal Down/Up blur ({s_du:.2e} rad), even with "
                f"max_compact_kernel_sz={caps[-1]}. This is a genuine limit of "
                f"the geometry: the compact kernel will be nearly pointwise and "
                f"the effective blur is dominated by Down/Up.",
                RuntimeWarning,
            )
        self.layer = layer

        # --- Compact Gaussian kernel (residual blur, in quadrature) ---
        residual = np.sqrt(max(self.sigma**2 - s_du**2, 0.0))
        sigma_c = max(
            self._rad_to_px(residual, layer.coarse_nside), 1e-6
        )  # avoid division by zero
        k = layer.compact_kernel_sz
        if 3 * sigma_c > k // 2:
            warnings.warn(
                "Compact kernel too small: the Gaussian is truncated.",
                RuntimeWarning,
            )

        x = np.arange(k) - k // 2
        xx, yy = np.meshgrid(x, x)
        W = np.exp(-(xx**2 + yy**2) / (2 * sigma_c**2))
        W = (W / W.sum()).reshape(1, 1, k * k).astype(np.float32)
        layer.set_compact_kernel(
            W, bias=np.zeros(1, np.float32), requires_grad=False
        )

        # Kept for diagnostics / debugging
        self.sigma_downup_rad = s_du
        self.sigma_compact_px = sigma_c
        self.max_compact_kernel_sz_used = cap

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
            return out.reshape(P)
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

    def check_impulse_response(self, centre_index=None):
        """Diagnostic: apply the filter to a Dirac and measure the effective sigma.

        Returns a dict with the measured sigma (radians), the ratio to the
        requested sigma, and the total mass of the response.
        """
        import healpix_geo.nested as hpg_n

        n = len(self.cell_ids)
        c = n // 2 if centre_index is None else int(centre_index)

        x = np.zeros(n, dtype=np.float32)
        x[c] = 1.0
        resp = np.asarray(self.predict(x)).ravel()

        lon, lat = hpg_n.healpix_to_lonlat(
            np.asarray(self.cell_ids), self.grid_info.level, ellipsoid=self.layer.ellipsoid
        )

        # drop float32 noise so it does not count as weight at large distances
        resp_clean = np.clip(resp, 0.0, None)
        resp_clean[resp_clean < 1e-6 * resp_clean.max()] = 0.0

        s_measured = self._effective_sigma(resp_clean, lon, lat, c)
        return {
            "sigma_requested": self.sigma,
            "sigma_measured": s_measured,
            "ratio": s_measured / self.sigma,
            "mass": float(resp.sum()),
            "sigma_downup": self.sigma_downup_rad,
            "n_levels": self.layer.n_levels,
            "cap_used": self.max_compact_kernel_sz_used,
        }


@dataclass
class FFTGaussian1DHealpix(Predictor):
    """Gaussian filter on a HEALPix patch, using a tangent-plane FFT convolution."""

    cell_ids: _ArrayLike
    grid_info: DGGSInfo

    sigma: float  # standard deviation, in radians
    kernel_size: int | None = None
    max_kernel_size: int | None = 129  # None = no cap (the FFT absorbs the cost)
    pixel_size_rad: float | None = None  # projected pixel size; None = nominal
    ellipsoid: str | None = None
    max_patch_radius_deg: float = 10.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self):
        from healpix_analyse import HealPixFFTConv

        level = self.grid_info.level
        nside = 2**level
        np_dtype = np.float64 if self.dtype == torch.float64 else np.float32
        if self.ellipsoid is None:
            self.ellipsoid = self.cell_ids.attrs.get("ellipsoid", "sphere")

        # Projected pixel size, passed explicitly to LocalFFT so that the
        # radians -> pixels conversion is exact in the projected plane.
        if self.pixel_size_rad is None:
            self.pixel_size_rad = np.sqrt(np.pi / 3) / nside  # nominal pixel side

        # sigma in projected pixels (no sqrt(pi) factor: pixel size is explicit)
        sigma_px = self.sigma / self.pixel_size_rad
        if sigma_px < 0.5:
            warnings.warn(
                f"sigma ({self.sigma:.2e} rad) is below half a pixel "
                f"({0.5 * self.pixel_size_rad:.2e} rad): the filter is nearly a Dirac.",
                RuntimeWarning,
            )

        # Truncation at 3 sigma
        radius = int(np.ceil(3 * sigma_px))
        kernel_size = 2 * radius + 1
        if self.max_kernel_size is not None and kernel_size > self.max_kernel_size:
            warnings.warn(
                f"kernel_size truncated from {kernel_size} to {self.max_kernel_size} "
                f"(sigma_px={sigma_px:.2f}): the effective sigma will be smaller "
                f"than requested.",
                RuntimeWarning,
            )
            kernel_size = self.max_kernel_size
            kernel_size -= 1 - kernel_size % 2  # keep it odd
        self.kernel_size = kernel_size

        self.layer = HealPixFFTConv(
            level,
            1,
            1,
            self.kernel_size,
            cell_ids=self.cell_ids,
            ellipsoid=self.ellipsoid,
            max_patch_radius_deg=self.max_patch_radius_deg,
            pixel_size_rad=self.pixel_size_rad,
            device=self.device,
            dtype=self.dtype,
        )

        # Gaussian kernel: exp(-r^2 / (2 sigma^2)), normalised to sum 1
        x = np.arange(self.kernel_size) - self.kernel_size // 2
        xx, yy = np.meshgrid(x, x)
        W = np.exp(-(xx**2 + yy**2) / (2 * max(sigma_px, 1e-6) ** 2))
        W = (W / W.sum()).astype(np_dtype).reshape(1, 1, self.kernel_size**2)

        self.layer.set_kernel(W, bias=np.zeros(1, np_dtype), requires_grad=False)
        self.sigma_px = sigma_px  # kept for diagnostics

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
            return out.reshape(P)
        if kind == "2d_bp":
            return out.reshape(orig_shape[0], orig_shape[1])
        if kind == "2d_fallback":
            return out.reshape(1, P)
        return out

    # def predict(self, X, mask=None):
    #     bcp, original_info = self._ensure_bcp(X)
    #     with torch.no_grad():
    #         out = self.layer(bcp.astype(np.float32))
    #     out_np = out if isinstance(out, np.ndarray) else out.detach().cpu().numpy()

    #     filtered = self._restore_shape(out_np, original_info)
    #     if mask is not None:
    #         filtered = np.where(mask, filtered, 0)
    #     return filtered
    def predict(self, X, mask=None):
        bcp, original_info = self._ensure_bcp(X)
        with torch.no_grad():
            out = self.layer(bcp)                       # pas de cast float32
        out_np = out if isinstance(out, np.ndarray) else out.detach().cpu().numpy()
        filtered = self._restore_shape(out_np, original_info)

        filtered = np.maximum(filtered, 0.0)            # bruit négatif de la FFT
        filtered[filtered < 1e-12* filtered.max()] = 0.0   # bruit relatif, à ajuster
        if mask is not None:
            filtered = np.where(mask, filtered, 0)
        return filtered