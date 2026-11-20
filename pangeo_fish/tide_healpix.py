"""Tidal evidence from harmonic constants stored on HEALPix cells.

The tide is predicted directly on the cells used by the HMM from tidal harmonic
constants, instead of computing a likelihood on a latitude/longitude grid and
regridding it.

Conventions
-----------
- The constants of constituent ``k`` are stored as one complex number per cell,
  ``C_k = A_k exp(+i G_k)``, with ``A_k`` the amplitude in metres and ``G_k`` the
  Greenwich phase lag. They are stored as two real variables, ``tide_real`` and
  ``tide_imag``, with dimensions ``(constituent, cells)``.
- The elevation is ``eta(t) = sum_k f_k(t) A_k cos(V_k(t) + u_k(t) - G_k)``, where
  ``V_k`` is the Greenwich astronomical argument and ``f_k``, ``u_k`` the nodal
  corrections. ``f`` and ``V + u`` are computed by pyfes (Darwin engine,
  Schureman formulae), which is an optional dependency.
- Minor constituents are not inferred: only the stored constituents are summed.
- ``cell_ids`` carries the DGGS attributes ``grid_name``, ``level``,
  ``indexing_scheme`` and ``ellipsoid``. The ellipsoid is required: the cell
  centres are computed with it (``"WGS84"`` gives geodetic latitudes, the
  latitudes of the tide models).

Contents
--------
- containers: :func:`harmonic_constants_dataset`, :func:`open_harmonic_constants`;
- prediction: :func:`nodal_modulations`, :func:`predict_tide`;
- evidence: :func:`still_tag_tide_likelihood` (with model-based tide detection),
  :func:`write_tide_pdf_healpix`;
- record analysis: :func:`harmonic_analysis`, :func:`constants_misfit`;
- storage: :func:`write_harmonic_constants`;
- interim builders from latitude/longitude sources (until the constants are
  produced by the GRID4EARTH preprocessing): :func:`latlon_constants_from_tmd`,
  :func:`latlon_constants_from_amplitude_phase`, :func:`constants_to_healpix`.
"""

import numpy as np
import pandas as pd
import xarray as xr

CONSTITUENT_DIM = "constituent"
CELL_DIM = "cells"
GRID_ATTRS = ("grid_name", "level", "indexing_scheme", "ellipsoid")
PHASE_CONVENTION = (
    "C = A exp(+iG), G Greenwich phase lag; eta = sum_k f_k A_k cos(V_k + u_k - G_k)"
)
TMD_EPOCH = pd.Timestamp("1992-01-01T00:00:00")


# ----------------------------------------------------------------------------
# grid helpers
# ----------------------------------------------------------------------------


def _grid_info(cell_ids):
    """DGGS attributes of ``cell_ids`` (a DataArray), with the ellipsoid required."""
    attrs = dict(getattr(cell_ids, "attrs", {}))
    missing = [name for name in ("level", "ellipsoid") if attrs.get(name) is None]
    if missing:
        raise ValueError(
            f"cell_ids must carry the DGGS attributes {missing}: set them explicitly"
            " (e.g. ellipsoid='WGS84'), the tide is sampled at the cell centres"
        )
    info = {
        "grid_name": attrs.get("grid_name", "healpix"),
        "level": int(attrs["level"]),
        "indexing_scheme": attrs.get("indexing_scheme", "nested"),
        "ellipsoid": attrs["ellipsoid"],
    }
    if info["grid_name"] != "healpix" or info["indexing_scheme"] != "nested":
        raise ValueError(f"only nested HEALPix cells are supported, got {info}")
    return info


def make_cell_ids(cell_ids, level, ellipsoid, indexing_scheme="nested"):
    """Return ``cell_ids`` as a DataArray on ``cells`` with the DGGS attributes."""
    values = np.asarray(getattr(cell_ids, "values", cell_ids), dtype="uint64")
    attrs = {
        "grid_name": "healpix",
        "level": int(level),
        "indexing_scheme": indexing_scheme,
        "ellipsoid": ellipsoid,
    }
    return xr.DataArray(values, dims=CELL_DIM, attrs=attrs, name="cell_ids")


def cell_centers(cell_ids):
    """Longitude in [-180, 180) and latitude of the cell centres, on the cell ellipsoid."""
    from healpix_geo.nested import healpix_to_lonlat

    info = _grid_info(cell_ids)
    lon, lat = healpix_to_lonlat(
        np.asarray(cell_ids.values, dtype="uint64"),
        info["level"],
        ellipsoid=info["ellipsoid"],
    )
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0, np.asarray(lat)


def _check_same_grid(expected, actual):
    """Raise when two grid definitions disagree (level, ellipsoid, scheme)."""
    diff = {
        k: (expected[k], actual[k])
        for k in GRID_ATTRS
        if str(expected[k]).lower() != str(actual[k]).lower()
    }
    if diff:
        raise ValueError(f"HEALPix grid mismatch (expected, got): {diff}")


# ----------------------------------------------------------------------------
# containers
# ----------------------------------------------------------------------------


def harmonic_constants_dataset(constants, constituents, cell_ids, attrs=None):
    """Build a harmonic-constants dataset on HEALPix cells.

    Parameters
    ----------
    constants : array-like of complex, shape (constituent, cells)
        ``A exp(+iG)`` in metres. NaN marks cells without constants (land).
    constituents : sequence of str
        Constituent names (pyfes naming, case-insensitive, e.g. ``"M2"``, ``"2N2"``).
    cell_ids : xarray.DataArray
        Cell ids with the DGGS attributes (see :func:`make_cell_ids`).
    attrs : dict, optional
        Dataset attributes (source, version, ...).
    """
    info = _grid_info(cell_ids)
    constants = np.asarray(constants)
    if constants.shape != (len(constituents), cell_ids.size):
        raise ValueError(
            f"constants have shape {constants.shape}, expected"
            f" {(len(constituents), cell_ids.size)}"
        )
    dims = (CONSTITUENT_DIM, CELL_DIM)
    var_attrs = {"units": "m", "phase_convention": PHASE_CONVENTION}
    ds = xr.Dataset(
        {
            "tide_real": (dims, constants.real, var_attrs | {"long_name": "A cos G"}),
            "tide_imag": (dims, constants.imag, var_attrs | {"long_name": "A sin G"}),
        },
        coords={
            CONSTITUENT_DIM: (CONSTITUENT_DIM, [str(c) for c in constituents]),
            "cell_ids": (CELL_DIM, np.asarray(cell_ids.values, "uint64"), info),
        },
        attrs=dict(attrs or {}),
    )
    return ds


def open_harmonic_constants(
    store,
    cell_ids=None,
    constituents=None,
    storage_options=None,
    ellipsoid=None,
):
    """Open harmonic constants stored on HEALPix (Zarr) and select cells/constituents.

    Parameters
    ----------
    store : str, path or mapping
        Zarr store with ``tide_real``, ``tide_imag`` on ``(constituent, cells)``.
    cell_ids : xarray.DataArray or array-like, optional
        Cells to return, in this order. Cells absent from the store get NaN.
        When a DataArray with DGGS attributes is given, its grid must match the store.
    constituents : sequence of str, optional
        Constituents to keep (case-insensitive). Default: all.
    storage_options : dict, optional
        Passed to ``xarray.open_dataset``.
    ellipsoid : str, optional
        Expected ellipsoid; raises if the store was built on another one.
    """
    ds = xr.open_dataset(
        store, engine="zarr", chunks={}, storage_options=storage_options
    )
    return select_constants(ds, cell_ids, constituents, ellipsoid)


def select_constants(ds, cell_ids=None, constituents=None, ellipsoid=None):
    """Validate a constants dataset and select cells and constituents."""
    for name in ("tide_real", "tide_imag"):
        if name not in ds:
            raise ValueError(f"harmonic constants need the variable {name!r}")
    info = _grid_info(ds["cell_ids"])
    if ellipsoid is not None:
        _check_same_grid(info, info | {"ellipsoid": ellipsoid})
    if constituents is not None:
        names = {str(c).lower(): str(c) for c in ds[CONSTITUENT_DIM].values}
        missing = [c for c in constituents if str(c).lower() not in names]
        if missing:
            raise KeyError(f"constituents not in the dataset: {missing}")
        ds = ds.sel({CONSTITUENT_DIM: [names[str(c).lower()] for c in constituents]})
    if cell_ids is not None:
        if isinstance(cell_ids, xr.DataArray) and "level" in cell_ids.attrs:
            _check_same_grid(info, _grid_info(cell_ids))
        wanted = np.asarray(getattr(cell_ids, "values", cell_ids), dtype="uint64")
        pos = pd.Index(np.asarray(ds["cell_ids"].values)).get_indexer(wanted)
        found = xr.DataArray(pos >= 0, dims=CELL_DIM)
        ds = ds.isel({CELL_DIM: np.where(pos >= 0, pos, 0)})
        ds = ds.assign(
            {name: ds[name].where(found) for name in ("tide_real", "tide_imag")}
        )
        ds = ds.assign_coords(cell_ids=(CELL_DIM, wanted, info))
    return ds


def as_complex(constants):
    """Complex constants ``A exp(iG)`` as a DataArray (constituent, cells)."""
    return constants["tide_real"] + 1j * constants["tide_imag"]


# ----------------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------------


def _require_pyfes():
    try:
        import pyfes
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "pangeo_fish.tide_healpix needs pyfes for the astronomical arguments and"
            " nodal corrections: install it from conda-forge or PyPI ('pyfes')"
        ) from err
    return pyfes


def _pyfes_names(constituents):
    """Map names (any case) to pyfes names."""
    pyfes = _require_pyfes()
    known = {}
    for name in pyfes.known_constituents():
        known.setdefault(name.lower(), name)
    out = []
    for c in constituents:
        key = str(c).strip().lower()
        if key not in known:
            raise ValueError(f"constituent {c!r} is unknown to pyfes")
        out.append(known[key])
    return out


def _to_utc_datetime64(times):
    times = pd.DatetimeIndex(np.atleast_1d(times))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    return times.values.astype("datetime64[us]")


def nodal_modulations(times, constituents):
    """Nodal factor ``f`` and phase ``V + u`` (radians) of each constituent.

    Parameters
    ----------
    times : array-like of datetime64 or DatetimeIndex
        UTC times (timezone-aware times are converted to UTC).
    constituents : sequence of str

    Returns
    -------
    xarray.Dataset
        ``f`` and ``vu`` with dimensions ``(time, constituent)``, constituents in
        the given order.
    """
    pyfes = _require_pyfes()
    names = _pyfes_names(constituents)
    epoch = _to_utc_datetime64(times)
    table = pyfes.wave_table_factory(pyfes.EngineType.DARWIN, names)
    order = list(table.constituents)  # pyfes sorts the constituents
    f, vu = table.compute_nodal_modulations(epoch)
    rows = [order.index(n) for n in names]
    dims = ("time", CONSTITUENT_DIM)
    coords = {
        "time": epoch.astype("datetime64[ns]"),
        CONSTITUENT_DIM: list(constituents),
    }
    return xr.Dataset(
        {"f": (dims, f[rows].T), "vu": (dims, vu[rows].T, {"units": "rad"})},
        coords=coords,
    )


def predict_tide(constants, times=None, modulations=None):
    """Tide elevation (m) at the given times on the cells of ``constants``.

    ``eta[t, c] = sum_k f[t, k] (Re C[k, c] cos vu[t, k] + Im C[k, c] sin vu[t, k])``.
    The prediction at a time only depends on that time (no reference to the start
    of the record). Cells with NaN constants give NaN.

    Parameters
    ----------
    constants : xarray.Dataset
        Output of :func:`open_harmonic_constants` or :func:`harmonic_constants_dataset`.
    times : array-like of datetime64, optional
        UTC times. Required unless ``modulations`` is given.
    modulations : xarray.Dataset, optional
        Precomputed output of :func:`nodal_modulations` (then pyfes is not needed).
    """
    if modulations is None:
        if times is None:
            raise ValueError("give times or modulations")
        modulations = nodal_modulations(times, list(constants[CONSTITUENT_DIM].values))
    else:
        modulations = modulations.sel(
            {CONSTITUENT_DIM: list(constants[CONSTITUENT_DIM].values)}
        )
    fcos = modulations["f"] * np.cos(modulations["vu"])
    fsin = modulations["f"] * np.sin(modulations["vu"])
    eta = xr.dot(fcos, constants["tide_real"], dim=CONSTITUENT_DIM) + xr.dot(
        fsin, constants["tide_imag"], dim=CONSTITUENT_DIM
    )
    eta = eta.transpose("time", CELL_DIM).rename("tide_elevation")
    eta.attrs = {"units": "m", "long_name": "tide elevation"}
    return eta.assign_coords(cell_ids=constants["cell_ids"])


# ----------------------------------------------------------------------------
# evidence for a still tag on the seabed
# ----------------------------------------------------------------------------


def _as_series(depth):
    if isinstance(depth, xr.DataArray):
        depth = depth.to_series()
    depth = pd.Series(depth).astype(float)
    index = pd.DatetimeIndex(depth.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return pd.Series(depth.values, index=index).sort_index()


def _seconds_since(index, t0):
    """Seconds from ``t0`` as floats (integer nanoseconds, no timedelta division)."""
    ns = pd.DatetimeIndex(index).as_unit("ns").asi8
    return (ns - pd.Timestamp(t0).as_unit("ns").value) / 1e9


def _bin_average(series, freq):
    """Average over ``freq`` intervals, each mean at the mean time of its samples.

    pandas labels a resampled mean with the start of its interval; predicting the
    tide at that label would shift it by half an interval (15 degrees of M2 for
    hourly means). The mean sample time is exact for any sampling (e.g. 10 min
    samples at :00..:50 average at :25). ``freq=None`` keeps the samples as they are.
    """
    series = series.dropna()
    if freq is None:
        return series
    t0 = series.index[0]
    seconds = pd.Series(_seconds_since(series.index, t0), index=series.index)
    values = series.resample(freq).mean()
    keep = values.notna().values
    mean_seconds = seconds.resample(freq).mean().values[keep]
    index = pd.DatetimeIndex(t0 + pd.to_timedelta(mean_seconds, unit="s"))
    return pd.Series(values.values[keep], index=index)


def still_tag_tide_likelihood(
    depth,
    constants,
    *,
    sigma,
    water_column=None,
    valid=None,
    freq="1h",
    time_step="1D",
    n_eff=None,
    ocean_mask=None,
    min_samples=12,
    detection_ratio=None,
    min_tidal_signal=0.05,
):
    """Tidal likelihood of a tag lying still on the seabed, on HEALPix cells.

    Model for the depth record averaged over ``freq`` (hourly by default)::

        d(t) = D_c + eta_c(t) + e(t),   e ~ N(0, sigma^2)

    with ``eta_c`` the tide predicted at cell ``c`` and ``D_c`` the mean water depth
    at the cell. When ``water_column`` is given it is used as ``D_c`` (depth and tide
    evidence together); otherwise ``D_c`` is estimated in each window by the mean of
    ``d - eta_c`` (profile likelihood: only the shape of the tide is used, combine it
    with a separate bathymetry likelihood).

    ``sigma`` is the model error (tide model, surge, sensor), not the variability of
    the depth within an hour. For each window of length ``time_step``::

        log L_c = -0.5 * n_eff * mean(r_c^2) / sigma^2

    with ``r_c`` the residuals. ``n_eff`` defaults to the number of samples of the
    window (independent errors). The tide model errors are correlated in time, so a
    smaller ``n_eff`` tempers the evidence.

    Parameters
    ----------
    depth : pandas.Series or xarray.DataArray
        Pressure depth of the tag (m, positive down) indexed by time (UTC).
    constants : xarray.Dataset
        Harmonic constants on the HMM cells.
    sigma : float
        Model error (m) on one ``freq`` sample.
    water_column : array-like or xarray.DataArray, optional
        Mean water depth (m, positive) per cell, same bathymetry as the depth evidence.
    valid : pandas.Series of bool, optional
        True where the tag is still on the seabed. Other samples are ignored.
    freq : str or None, default "1h"
        Averaging period of the depth record; each mean is attributed to the mean
        time of its samples. None: use the samples as they are.
    time_step : str or None, default "1D"
        Window of the likelihood (the HMM time step). None: one window for the whole
        record.
    n_eff : float, optional
        Effective number of independent samples per window.
    ocean_mask : array-like of bool, optional
        Cells allowed (in addition to the cells with constants).
    min_samples : int, default 12
        Windows with fewer valid samples get a uniform pdf.
    detection_ratio : float, optional
        Model-based tide detection. A window has a tide when the best cell explains
        it: ``min_c rms_c < detection_ratio * no_tide_rms``, with ``no_tide_rms`` the
        RMS of the record around its window mean, and ``no_tide_rms >=
        min_tidal_signal``. Windows without a tide get a uniform pdf. This works for
        semidiurnal, mixed and diurnal regimes alike (no frequency band or amplitude
        threshold). Default: no detection, every window with enough samples is used.
    min_tidal_signal : float, default 0.05
        Minimum ``no_tide_rms`` (m) for a detection (a flat record has no tide).

    Returns
    -------
    xarray.Dataset
        ``pdf`` (time, cells), normalised over the allowed cells, ``log_likelihood``,
        ``rms`` (residual RMS, m), ``n_samples``, ``no_tide_rms``, ``best_rms`` and
        ``tide_found`` (time, 1 when the window is used as evidence), with
        ``cell_ids`` and its grid attributes. ``time`` is the start of each window.
    """
    series = _as_series(depth)
    record_start, record_end = series.index.min(), series.index.max()
    if valid is not None:
        valid = _as_series(valid).reindex(series.index).fillna(0).astype(bool)
        series = series[valid]
    hourly = _bin_average(series, freq)

    n_cells = constants.sizes[CELL_DIM]
    allowed = np.isfinite(constants["tide_real"]).all(CONSTITUENT_DIM).values
    if ocean_mask is not None:
        allowed &= np.asarray(getattr(ocean_mask, "values", ocean_mask), dtype=bool)
    if water_column is not None:
        wc = np.asarray(getattr(water_column, "values", water_column), dtype=float)
        allowed &= np.isfinite(wc)

    if time_step is None:
        starts = pd.DatetimeIndex([record_start.floor("D")])
        labels = np.zeros(len(hourly), dtype=int)
    else:
        first = record_start.floor(time_step)
        last = record_end.floor(time_step)
        starts = pd.date_range(first, last, freq=time_step)
        labels = starts.get_indexer(hourly.index.floor(time_step))

    n_win = len(starts)
    loglik = np.full((n_win, n_cells), np.nan)
    rms = np.full((n_win, n_cells), np.nan)
    n_samples = np.zeros(n_win, dtype=int)
    no_tide_rms = np.full(n_win, np.nan)
    for w in range(n_win):
        sel = hourly[labels == w]
        n_samples[w] = len(sel)
        if len(sel) < max(min_samples, 1):
            continue
        no_tide_rms[w] = float(np.sqrt(((sel - sel.mean()) ** 2).mean()))
        eta = predict_tide(constants, sel.index.values).values  # (time, cells)
        r = sel.values[:, None] - eta
        if water_column is None:
            r = r - r.mean(axis=0, keepdims=True)
        else:
            r = r - wc[None, :]
        ms = (r**2).mean(axis=0)
        n = len(sel) if n_eff is None else n_eff
        loglik[w] = -0.5 * n * ms / sigma**2
        rms[w] = np.sqrt(ms)

    best_rms = np.full(n_win, np.nan)
    for w in range(n_win):
        r = np.where(allowed, rms[w], np.nan)
        if np.isfinite(r).any():
            best_rms[w] = np.nanmin(r)
    used = np.isfinite(best_rms)
    if detection_ratio is not None:
        used &= (best_rms < detection_ratio * no_tide_rms) & (
            no_tide_rms >= min_tidal_signal
        )

    pdf = np.zeros((n_win, n_cells))
    uniform = allowed / max(allowed.sum(), 1)
    for w in range(n_win):
        ll = np.where(allowed, loglik[w], np.nan)
        if not used[w] or not np.isfinite(ll).any():
            pdf[w] = uniform
            continue
        p = np.where(np.isfinite(ll), np.exp(ll - np.nanmax(ll)), 0.0)
        pdf[w] = p / p.sum()

    tide_found = used.astype(int)
    dims = ("time", CELL_DIM)
    ds = xr.Dataset(
        {
            "pdf": (dims, pdf),
            "log_likelihood": (dims, loglik),
            "rms": (dims, rms, {"units": "m"}),
            "n_samples": ("time", n_samples),
            "no_tide_rms": ("time", no_tide_rms, {"units": "m"}),
            "best_rms": ("time", best_rms, {"units": "m"}),
            "tide_found": ("time", tide_found),
            "ocean_mask": (CELL_DIM, allowed),
        },
        coords={"time": starts.values, "cell_ids": constants["cell_ids"]},
        attrs={
            "sigma_m": float(sigma),
            "n_eff": "n_samples" if n_eff is None else float(n_eff),
            "depth_reference": "water_column" if water_column is not None else "mean",
            "freq": freq,
            "time_step": "segment" if time_step is None else time_step,
            "detection_ratio": (
                "none" if detection_ratio is None else float(detection_ratio)
            ),
        },
    )
    return ds


def _frequencies_rad_per_hour(constituents):
    pyfes = _require_pyfes()
    names = _pyfes_names(constituents)
    table = pyfes.wave_table_factory(pyfes.EngineType.DARWIN, names)
    return np.array([table[n].frequency(pyfes.RADIAN_PER_HOUR) for n in names])


def harmonic_analysis(depth, constituents=("M2", "S2", "K1", "O1"), freq="1h"):
    """Least-squares harmonic constants of a depth record (Greenwich convention).

    Fits ``d(t) = D + sum_k f_k(t) (a_k cos(V_k + u_k) + b_k sin(V_k + u_k))`` with the
    pyfes arguments, so that ``a_k + i b_k = A_k exp(+iG_k)`` is directly comparable
    with the harmonic constants of a tide model.

    Parameters
    ----------
    depth : pandas.Series or xarray.DataArray
        Depth (m, positive down) indexed by UTC time.
    constituents : sequence of str
        Constituents to fit. Pairs that the record cannot separate (Rayleigh
        criterion: frequency difference times record length < one cycle) raise.
    freq : str or None, default "1h"
        Averaging period applied before the fit (means at the mean sample time).

    Returns
    -------
    xarray.Dataset
        ``tide_real``, ``tide_imag`` (constituent), ``amplitude`` (m), ``phase``
        (degrees, Greenwich lag), with attributes ``mean_depth``, ``residual_rms``
        and ``n_samples``.
    """
    series = _bin_average(_as_series(depth), freq)
    constituents = list(constituents)
    omega = _frequencies_rad_per_hour(constituents)
    hours = (series.index.max() - series.index.min()).total_seconds() / 3600.0
    for i in range(len(omega)):
        for j in range(i + 1, len(omega)):
            if abs(omega[i] - omega[j]) * hours < 2 * np.pi:
                raise ValueError(
                    f"{constituents[i]} and {constituents[j]} cannot be separated"
                    f" with {hours / 24:.1f} days of data (Rayleigh criterion)"
                )
    mod = nodal_modulations(series.index.values, constituents)
    f, vu = mod["f"].values, mod["vu"].values
    design = np.column_stack([np.ones(len(series)), f * np.cos(vu), f * np.sin(vu)])
    coef, *_ = np.linalg.lstsq(design, series.values, rcond=None)
    n = len(constituents)
    c = coef[1 : 1 + n] + 1j * coef[1 + n :]
    residual = series.values - design @ coef
    dims = CONSTITUENT_DIM
    return xr.Dataset(
        {
            "tide_real": (dims, c.real, {"units": "m"}),
            "tide_imag": (dims, c.imag, {"units": "m"}),
            "amplitude": (dims, np.abs(c), {"units": "m"}),
            "phase": (dims, np.degrees(np.angle(c)) % 360, {"units": "degree"}),
        },
        coords={CONSTITUENT_DIM: constituents},
        attrs={
            "mean_depth": float(coef[0]),
            "residual_rms": float(np.sqrt(np.mean(residual**2))),
            "n_samples": len(series),
            "phase_convention": PHASE_CONVENTION,
        },
    )


def constants_misfit(observed, constants):
    """Distance (m) between observed constants and the constants of each cell.

    ``sqrt(sum_k |C_obs,k - C_cell,k|^2)`` over the constituents of ``observed``
    (e.g. the output of :func:`harmonic_analysis`). Works on HEALPix constants
    (``cells``) and on latitude/longitude constants alike.
    """
    names = {str(c).lower(): str(c) for c in constants[CONSTITUENT_DIM].values}
    missing = [
        c for c in observed[CONSTITUENT_DIM].values if str(c).lower() not in names
    ]
    if missing:
        raise KeyError(f"constituents not in the model constants: {missing}")
    model = constants.sel(
        {
            CONSTITUENT_DIM: [
                names[str(c).lower()] for c in observed[CONSTITUENT_DIM].values
            ]
        }
    )
    obs_c = (observed["tide_real"] + 1j * observed["tide_imag"]).values
    mod_c = (model["tide_real"] + 1j * model["tide_imag"]).values
    shape = (-1,) + (1,) * (mod_c.ndim - 1)
    dist = np.sqrt((np.abs(mod_c - obs_c.reshape(shape)) ** 2).sum(axis=0))
    other = [d for d in model["tide_real"].dims if d != CONSTITUENT_DIM]
    coords = {k: v for k, v in model.coords.items() if k != CONSTITUENT_DIM}
    return xr.DataArray(dist, dims=other, coords=coords, name="constants_misfit")


def write_harmonic_constants(constants, store, chunk_cells=2**16, dtype="float32"):
    """Write harmonic constants to Zarr: all constituents in one chunk, cells chunked.

    Parameters
    ----------
    constants : xarray.Dataset
        Output of :func:`harmonic_constants_dataset` or :func:`constants_to_healpix`.
    store : str or path
    chunk_cells : int
        Cells per chunk (contiguous nested ranges when the cells are sorted).
    dtype : str
        Storage type of ``tide_real`` and ``tide_imag`` (float32: < 1e-7 m error per metre).
    """
    from pangeo_fish.storage import _with_dggs_attrs

    _grid_info(constants["cell_ids"])
    order = np.argsort(constants["cell_ids"].values, kind="stable")
    ds = constants.isel({CELL_DIM: order})
    ds = ds.assign(
        {name: ds[name].astype(dtype) for name in ("tide_real", "tide_imag")}
    )
    ds = _with_dggs_attrs(ds).chunk({CONSTITUENT_DIM: -1, CELL_DIM: chunk_cells})
    ds.to_zarr(store, mode="w", zarr_format=2, consolidated=True)
    return store


def write_tide_pdf_healpix(likelihood, target_root, behav_found=None, step=200):
    """Write ``tide_pdf_healpix.zarr`` in the layout read by ``merge_tide_with_total``.

    Parameters
    ----------
    likelihood : xarray.Dataset
        Output of :func:`still_tag_tide_likelihood` with a daily ``time_step``.
    target_root : str or Path
        Folder receiving ``tide_pdf_healpix.zarr``.
    behav_found : array-like, optional
        Daily behaviour flag; zeros by default.
    """
    import zarr

    from pangeo_fish.storage import _with_dggs_attrs, write_time_blocks

    ds = likelihood[["pdf", "ocean_mask", "tide_found"]].copy()
    flags = np.zeros(ds.sizes["time"]) if behav_found is None else behav_found
    ds["behav_found"] = ("time", np.asarray(flags, dtype=float))
    ds["tide_found"] = ds["tide_found"].astype(float)
    ds = _with_dggs_attrs(ds)
    store = f"{target_root}/tide_pdf_healpix.zarr"
    write_time_blocks(ds.chunk({"cells": -1}), store, step=step)
    zarr.consolidate_metadata(store)
    return store


# ----------------------------------------------------------------------------
# interim builders: latitude/longitude constants -> HEALPix cells
# ----------------------------------------------------------------------------


def _latlon_constants(constants, constituents, lat, lon, mask, attrs):
    """Standard lat/lon constants dataset (constituent, lat, lon), lon in [-180, 180)."""
    lon = (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0
    lat = np.asarray(lat, dtype=float)
    ilon = np.argsort(lon, kind="stable")
    ilat = np.argsort(lat, kind="stable")
    constants = np.asarray(constants)[:, ilat][:, :, ilon]
    mask = np.asarray(mask, dtype=bool)[ilat][:, ilon]
    mask &= np.isfinite(constants).all(axis=0)
    dims = (CONSTITUENT_DIM, "lat", "lon")
    return xr.Dataset(
        {
            "tide_real": (dims, constants.real),
            "tide_imag": (dims, constants.imag),
            "mask": (("lat", "lon"), mask),
        },
        coords={
            CONSTITUENT_DIM: list(constituents),
            "lat": lat[ilat],
            "lon": lon[ilon],
        },
        attrs=dict(attrs) | {"phase_convention": PHASE_CONVENTION},
    )


def _bbox_slices(lat, lon, bbox, margin):
    if bbox is None:
        return slice(None), slice(None)
    lon = (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0
    lat_min, lat_max = bbox["latitude"]
    lon_min, lon_max = bbox["longitude"]
    ilat = np.flatnonzero((lat >= lat_min - margin) & (lat <= lat_max + margin))
    ilon = np.flatnonzero((lon >= lon_min - margin) & (lon <= lon_max + margin))
    return ilat, ilon


def latlon_constants_from_tmd(source, bbox=None, constituents=None, margin=0.1):
    """Read a TMD3 netCDF file (e.g. TPXO9-atlas, Greene's conversion) as lat/lon constants.

    The file stores ``hRe + i hIm = A exp(+iG')`` with ``G'`` relative to its own
    astronomical arguments ``phase`` (at 1992-01-01) and frequencies ``omega``. The
    phases are re-referenced to the pyfes Greenwich arguments:
    ``C = (hRe + i hIm) exp(i (V_pyfes(1992-01-01) - phase))``. For the TPXO9 file
    this shift is zero except for S1 (180 degrees, a convention difference).

    Parameters
    ----------
    source : str, path or xarray.Dataset
    bbox : dict, optional
        ``{"latitude": [min, max], "longitude": [min, max]}`` (longitude in -180..180).
    constituents : sequence of str, optional
    margin : float
        Extra degrees around the bbox (for the interpolation).
    """
    ds = xr.open_dataset(source) if not isinstance(source, xr.Dataset) else source
    names = ds["constituents"].attrs["constituent_order"].split()
    keep = range(len(names))
    if constituents is not None:
        lower = [n.lower() for n in names]
        keep = [lower.index(str(c).lower()) for c in constituents]
    names = [names[k] for k in keep]
    ilat, ilon = _bbox_slices(ds["lat"].values, ds["lon"].values, bbox, margin)
    sub = ds.isel(lat=ilat, lon=ilon, constituents=list(keep))

    phase0 = ds["phase"].values[list(keep), 0].astype(float)
    pyfes = _require_pyfes()
    table = pyfes.wave_table_factory(pyfes.EngineType.DARWIN, _pyfes_names(names))
    angles = pyfes.AstronomicAngle(pyfes.Formulae.SCHUREMAN_ORDER_1)
    angles.update(TMD_EPOCH.to_pydatetime())
    table.compute_nodal_corrections(angles)
    v0 = np.array([table[n].v for n in _pyfes_names(names)])
    shift = np.angle(np.exp(1j * (v0 - phase0)))

    hc = sub["hRe"].values + 1j * sub["hIm"].values  # (k, lat, lon)
    hc = hc * np.exp(1j * shift)[:, None, None]
    attrs = {
        "source": ds.attrs.get("Title", "TMD3 file"),
        "phase_shift_to_pyfes_deg": np.degrees(shift).round(4).tolist(),
    }
    return _latlon_constants(
        hc, names, sub["lat"].values, sub["lon"].values, sub["mask"].values > 0, attrs
    )


def latlon_constants_from_amplitude_phase(
    amplitude, phase, constituents, lat, lon, mask=None, amplitude_scale=1.0, attrs=None
):
    """Lat/lon constants from amplitude and Greenwich phase lag (FES-style inputs).

    Parameters
    ----------
    amplitude, phase : array-like, shape (constituent, lat, lon)
        Amplitude (multiplied by ``amplitude_scale`` to get metres; FES files are in
        cm, use 0.01) and Greenwich phase lag in degrees. Missing values are NaN.
    constituents : sequence of str
    lat, lon : array-like
    mask : array-like of bool, shape (lat, lon), optional
        Ocean mask; default: where all constituents are finite.
    """
    amplitude = np.asarray(amplitude, dtype=float) * amplitude_scale
    phase = np.radians(np.asarray(phase, dtype=float))
    constants = amplitude * np.exp(1j * phase)
    if mask is None:
        mask = np.isfinite(constants).all(axis=0)
    return _latlon_constants(constants, constituents, lat, lon, mask, attrs or {})


def _bilinear_weights(axis, x):
    """Lower index and weight of the upper point for points x on a sorted axis."""
    i0 = np.searchsorted(axis, x, side="right") - 1
    inside = (i0 >= 0) & (i0 < len(axis) - 1)
    inside |= x == axis[-1]
    i0 = np.clip(i0, 0, len(axis) - 2)
    w = (x - axis[i0]) / (axis[i0 + 1] - axis[i0])
    return i0, np.clip(w, 0.0, 1.0), inside


def constants_to_healpix(latlon, cell_ids, fill_distance_km=None, attrs=None):
    """Resample lat/lon harmonic constants to HEALPix cell centres (complex values).

    The real and imaginary parts ``A cos G`` and ``A sin G`` are interpolated
    bilinearly (never amplitude and phase, which break at the phase wrap and near
    amphidromes). Land points of the source are left out and the weights are
    renormalised, so coastal cells use the neighbouring ocean points. This is meant
    for cells of about the source resolution or finer (bilinear); coarser cells
    need an area-weighted mean (GRID4EARTH preprocessing).

    Parameters
    ----------
    latlon : xarray.Dataset
        Output of :func:`latlon_constants_from_tmd` or
        :func:`latlon_constants_from_amplitude_phase`.
    cell_ids : xarray.DataArray
        Target cells with DGGS attributes (the ellipsoid decides the latitudes).
    fill_distance_km : float, optional
        Cells without any ocean source point among their 4 neighbours take the value
        of the nearest resampled cell within this distance (coastal gaps between the
        source land mask and the HMM ocean mask). Default: left NaN.
    """
    lon_c, lat_c = cell_centers(cell_ids)
    lat = latlon["lat"].values
    lon = latlon["lon"].values
    i0, wy, in_lat = _bilinear_weights(lat, lat_c)
    j0, wx, in_lon = _bilinear_weights(lon, lon_c)
    values = (latlon["tide_real"].values + 1j * latlon["tide_imag"].values).astype(
        complex
    )
    mask = latlon["mask"].values.astype(bool)
    values = np.where(mask[None], values, 0.0)

    total = np.zeros((values.shape[0], lat_c.size), dtype=complex)
    weight = np.zeros(lat_c.size)
    for di, wi in ((0, 1.0 - wy), (1, wy)):
        for dj, wj in ((0, 1.0 - wx), (1, wx)):
            w = wi * wj * mask[i0 + di, j0 + dj]
            total += values[:, i0 + di, j0 + dj] * w[None]
            weight += w
    ok = (weight > 1e-12) & in_lat & in_lon
    result = np.full_like(total, np.nan + 1j * np.nan)
    result[:, ok] = total[:, ok] / weight[ok][None]

    if fill_distance_km is not None and ok.any() and (~ok).any():
        from scipy.spatial import cKDTree

        def unit(lon_deg, lat_deg):
            lo, la = np.radians(lon_deg), np.radians(lat_deg)
            return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]

        tree = cKDTree(unit(lon_c[ok], lat_c[ok]))
        chord = 2 * np.sin(fill_distance_km / 6371.0 / 2)
        dist, idx = tree.query(unit(lon_c[~ok], lat_c[~ok]), distance_upper_bound=chord)
        found = np.isfinite(dist)
        targets = np.flatnonzero(~ok)[found]
        result[:, targets] = result[:, np.flatnonzero(ok)[idx[found]]]

    out_attrs = dict(latlon.attrs) | dict(attrs or {})
    out_attrs["resampling"] = "bilinear on complex values, land points excluded" + (
        f", nearest fill within {fill_distance_km} km" if fill_distance_km else ""
    )
    return harmonic_constants_dataset(
        result, list(latlon[CONSTITUENT_DIM].values), cell_ids, attrs=out_attrs
    )
