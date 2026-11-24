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


def _names(constituents):
    """Constituent names as an object array.

    A fixed-width or pandas string index cannot be encoded to Zarr by some
    xarray/pandas combinations (xarray 2025.7 with pandas 3); object arrays always can.
    """
    return np.array([str(c) for c in constituents], dtype=object)


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
            CONSTITUENT_DIM: (CONSTITUENT_DIM, _names(constituents)),
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
    *,
    require_all_cells=False,
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
    require_all_cells : bool, optional
        Raise if any requested cell ID is absent. Stored NaN values are preserved.
    """
    ds = xr.open_dataset(
        store, engine="zarr", chunks={}, storage_options=storage_options
    )
    try:
        selected = select_constants(
            ds, cell_ids, constituents, ellipsoid, require_all_cells=require_all_cells
        )
    except Exception:
        ds.close()
        raise
    if selected is not ds:
        selected.set_close(ds.close)
    return selected


def select_constants(
    ds, cell_ids=None, constituents=None, ellipsoid=None, *, require_all_cells=False
):
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
        missing = pos < 0
        if require_all_cells and missing.any():
            raise ValueError(
                f"Harmonic constants are missing {missing.sum()} requested HEALPix "
                f"cell IDs (first: {wanted[missing][0]})"
            )
        found = xr.DataArray(~missing, dims=CELL_DIM)
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
        CONSTITUENT_DIM: _names(constituents),
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
        available = {str(c).lower(): c for c in modulations[CONSTITUENT_DIM].values}
        wanted = [str(c) for c in constants[CONSTITUENT_DIM].values]
        missing = [c for c in wanted if c.lower() not in available]
        if missing:
            raise KeyError(f"constituents missing from the modulations: {missing}")
        modulations = modulations.sel(
            {CONSTITUENT_DIM: [available[c.lower()] for c in wanted]}
        ).assign_coords({CONSTITUENT_DIM: constants[CONSTITUENT_DIM].values})
    return _tide_from_basis(constants, _tidal_basis(modulations))


def _tidal_basis(modulations):
    f, vu = modulations["f"], modulations["vu"]
    return xr.Dataset({"fcos": f * np.cos(vu), "fsin": f * np.sin(vu)})


def _tide_from_basis(constants, basis):
    eta = xr.dot(basis["fcos"], constants["tide_real"], dim=CONSTITUENT_DIM) + xr.dot(
        basis["fsin"], constants["tide_imag"], dim=CONSTITUENT_DIM
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


def _step_ns(freq):
    """Length of ``freq`` (e.g. ``"1D"``, ``"12h"``) as ``numpy.timedelta64[ns]``.

    Built from integer nanoseconds: ``pd.Timedelta("12h")`` emits the NumPy
    generic-unit DeprecationWarning with recent NumPy/pandas (an error in CI).
    """
    t0 = pd.Timestamp("2000-01-01")
    ns = (t0 + pd.tseries.frequencies.to_offset(freq)).value - t0.value
    return np.timedelta64(int(ns), "ns")


def _seconds_since(index, t0):
    """Seconds from ``t0`` as floats (integer nanoseconds, no timedelta division)."""
    ns = pd.DatetimeIndex(index).as_unit("ns").asi8
    return (ns - pd.Timestamp(t0).as_unit("ns").value) / 1e9


def _bin_average(series, freq):
    """Average over ``freq`` intervals, each mean at the mean time of its samples.

    pandas labels a resampled mean with the start of its interval; predicting the
    tide at that label would shift it by half an interval (15 degrees of M2 for
    hourly means). The mean sample time is exact for any sampling (e.g. 10 min
    samples at :00..:50 average at :25). Tide predictions are averaged separately
    at those sample times. ``freq=None`` keeps the samples as they are.
    """
    series = series.dropna()
    if freq is None or series.empty:
        return series
    t0 = series.index[0]
    seconds = pd.Series(_seconds_since(series.index, t0), index=series.index)
    values = series.resample(freq).mean()
    keep = values.notna().values
    mean_seconds = seconds.resample(freq).mean().values[keep]
    index = pd.DatetimeIndex(t0 + pd.to_timedelta(mean_seconds, unit="s"))
    return pd.Series(values.values[keep], index=index)


def _bin_nodal_basis(series, binned, freq, constituents):
    """Average the harmonic basis at exactly the timestamps used for each depth mean."""
    basis = _tidal_basis(nodal_modulations(series.index.values, constituents))
    if freq is not None:
        # Use the original timestamps for bin membership, even if pyfes rounded them.
        basis = basis.assign_coords(time=series.index)
        counts = series.resample(freq).count()
        basis = basis.resample(time=freq).mean().sel(time=counts.index[counts > 0])
    return basis.assign_coords(time=binned.index)


def _per_cell(values, constants, name, dtype):
    """Per-cell values in the order of ``constants``: aligned on ``cell_ids`` when given."""
    cell_ids = constants["cell_ids"].values
    if isinstance(values, xr.DataArray) and "cell_ids" in values.coords:
        if values.dims != (CELL_DIM,):
            raise ValueError(f"{name} must have the single dimension {CELL_DIM!r}")
        if "level" in values["cell_ids"].attrs:
            _check_same_grid(
                _grid_info(constants["cell_ids"]), _grid_info(values["cell_ids"])
            )
        pos = pd.Index(np.asarray(values["cell_ids"].values)).get_indexer(cell_ids)
        if (pos < 0).any():
            raise ValueError(f"{name} lacks {int((pos < 0).sum())} cells of constants")
        return np.asarray(values.values[pos], dtype=dtype)
    out = np.asarray(getattr(values, "values", values), dtype=dtype)
    if out.shape != cell_ids.shape:
        raise ValueError(
            f"{name} has shape {out.shape}, constants have {cell_ids.size} cells;"
            " pass a DataArray with cell_ids to align by cell"
        )
    return out


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
    the depth within an hour. For each window of length ``time_step``, centred on its
    label ``T`` (``[T - 12 h, T + 12 h)`` for one day, like the temperature emission)::

        log L_c = -0.5 * n_eff * mean(r_c^2) / sigma^2

    with ``r_c`` the residuals. ``n_eff`` defaults to the number of samples of the
    window (independent errors). The tide model errors are correlated in time, so a
    smaller ``n_eff`` tempers the evidence. A partly covered window counts
    ``n_eff`` times its coverage (number of samples / samples of a full window).

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
        It must be referenced to mean sea level, the level around which the tide is
        predicted: charts referenced to the lowest astronomical tide (e.g. EMODnet
        DTM) are deeper by several metres in macrotidal areas and must be converted
        first. A DataArray with ``cell_ids`` is aligned on the cells of ``constants``;
        an array must follow their order.
    valid : pandas.Series of bool, optional
        True where the tag is still on the seabed. Other samples are ignored.
    freq : str or None, default "1h"
        Averaging period of the depth record and of the predicted tide at the same
        sample times. Each mean is attributed to the mean time of its samples.
        None: use the samples as they are.
    time_step : str or None, default "1D"
        Window of the likelihood (the HMM time step). None: one window for the whole
        record.
    n_eff : float, optional
        Effective number of independent samples per window.
    ocean_mask : array-like of bool, optional
        Cells allowed (in addition to the cells with constants). Aligned like
        ``water_column``.
    min_samples : int, default 12
        Windows with fewer valid samples get a uniform pdf.
    detection_ratio : float, optional
        Model-based tide detection. A window has a tide when the best cell explains
        it: ``min_c rms_c < detection_ratio * no_tide_rms``, with ``no_tide_rms`` the
        RMS of the record around its window mean, ``rms_c`` the RMS of the residuals
        around their window mean (also with ``water_column``, so that a bathymetry
        bias does not hide the tide) and ``no_tide_rms >= min_tidal_signal``. Windows without a tide get a uniform pdf. This works for
        semidiurnal, mixed and diurnal regimes alike (no frequency band or amplitude
        threshold). Default: no detection, every window with enough samples is used.
    min_tidal_signal : float, default 0.05
        Minimum ``no_tide_rms`` (m) for a detection (a flat record has no tide).

    Returns
    -------
    xarray.Dataset
        ``pdf`` (time, cells), normalised over the allowed cells, ``log_likelihood``,
        ``rms`` (residual RMS, m, including the depth offset when ``water_column`` is
        given), ``n_samples``, ``no_tide_rms``, ``best_rms`` (smallest RMS of the
        demeaned residuals over the allowed cells, the detection statistic) and
        ``tide_found`` (time, 1 when the window is used as evidence, else 0), with
        ``cell_ids`` and its grid attributes. ``time`` is the centre of each window.
        Windows without valid samples (e.g. ``valid`` all False) get a uniform pdf.

    Raises
    ------
    ValueError
        If the depth record is empty or no cell has finite constants and passes
        the supplied masks.
    """
    series = _as_series(depth).dropna()
    if series.empty:
        raise ValueError("the depth record has no samples")
    record_start, record_end = series.index.min(), series.index.max()
    # read the constants once (a Zarr-backed dataset would be read for every window)
    constants = constants.compute()
    if valid is not None:
        valid = _as_series(valid).reindex(series.index).fillna(0).astype(bool)
        series = series[valid]
    hourly = _bin_average(series, freq)

    n_cells = constants.sizes[CELL_DIM]
    allowed = (
        (np.isfinite(constants["tide_real"]) & np.isfinite(constants["tide_imag"]))
        .all(CONSTITUENT_DIM)
        .values
    )
    if ocean_mask is not None:
        allowed &= _per_cell(ocean_mask, constants, "ocean_mask", bool)
    if water_column is not None:
        wc = _per_cell(water_column, constants, "water_column", float)
        allowed &= np.isfinite(wc)
    if not allowed.any():
        raise ValueError("no eligible cells with finite tide constants")

    if time_step is None:
        starts = pd.DatetimeIndex([record_start.floor("D")])
        labels = np.zeros(len(hourly), dtype=int)
    else:
        # windows centred on their label T: [T - time_step/2, T + time_step/2),
        # the convention of the temperature emission (cf_xarray time bounds)
        half = _step_ns(time_step) / 2
        first = (record_start + half).floor(time_step)
        last = (record_end + half).floor(time_step)
        starts = pd.date_range(first, last, freq=time_step)
        labels = starts.get_indexer((hourly.index + half).floor(time_step))
    if time_step is not None and freq is not None:
        # integer nanoseconds: no timedelta division (NumPy generic-unit deprecation)
        window = np.diff(pd.date_range("2000-01-01", periods=2, freq=time_step).asi8)
        sample = np.diff(pd.date_range("2000-01-01", periods=2, freq=freq).asi8)
        full_window = float(window[0] / sample[0])
    else:
        full_window = None

    n_win = len(starts)
    loglik = np.full((n_win, n_cells), np.nan)
    rms = np.full((n_win, n_cells), np.nan)
    n_samples = np.bincount(labels[labels >= 0], minlength=n_win)
    no_tide_rms = np.full(n_win, np.nan)
    shape_rms = np.full((n_win, n_cells), np.nan)
    eligible_windows = np.flatnonzero(n_samples >= max(min_samples, 1))
    if eligible_windows.size:
        basis = _bin_nodal_basis(
            series, hourly, freq, constants[CONSTITUENT_DIM].values
        )
    for w in eligible_windows:
        in_window = labels == w
        sel = hourly[in_window]
        no_tide_rms[w] = float(np.sqrt(((sel - sel.mean()) ** 2).mean()))
        eta = _tide_from_basis(constants, basis.isel(time=in_window)).values
        r = sel.values[:, None] - eta
        demeaned = r - r.mean(axis=0, keepdims=True)
        shape_rms[w] = np.sqrt((demeaned**2).mean(axis=0))
        r = demeaned if water_column is None else r - wc[None, :]
        ms = (r**2).mean(axis=0)
        if n_eff is None:
            n = len(sel)
        elif full_window is None:
            n = n_eff
        else:
            n = n_eff * min(1.0, len(sel) / full_window)
        loglik[w] = -0.5 * n * ms / sigma**2
        rms[w] = np.sqrt(ms)

    best_rms = np.full(n_win, np.nan)
    for w in range(n_win):
        r = np.where(allowed, shape_rms[w], np.nan)
        if np.isfinite(r).any():
            best_rms[w] = np.nanmin(r)
    used = np.isfinite(best_rms)
    if detection_ratio is not None:
        used &= (best_rms < detection_ratio * no_tide_rms) & (
            no_tide_rms >= min_tidal_signal
        )

    pdf = np.zeros((n_win, n_cells))
    uniform = allowed / allowed.sum()
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
        Averaging period applied to both depth and the harmonic basis before the fit.

    Returns
    -------
    xarray.Dataset
        ``tide_real``, ``tide_imag`` (constituent), ``amplitude`` (m), ``phase``
        (degrees, Greenwich lag), with attributes ``mean_depth``, ``residual_rms``
        and ``n_samples``.
    """
    raw = _as_series(depth).dropna()
    series = _bin_average(raw, freq)
    constituents = list(constituents)
    omega = _frequencies_rad_per_hour(constituents)
    hours = float(np.ptp(_seconds_since(series.index, series.index[0]))) / 3600.0
    for i in range(len(omega)):
        for j in range(i + 1, len(omega)):
            if abs(omega[i] - omega[j]) * hours < 2 * np.pi:
                raise ValueError(
                    f"{constituents[i]} and {constituents[j]} cannot be separated"
                    f" with {hours / 24:.1f} days of data (Rayleigh criterion)"
                )
    basis = _bin_nodal_basis(raw, series, freq, constituents)
    design = np.column_stack(
        [np.ones(len(series)), basis["fcos"].values, basis["fsin"].values]
    )
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
        coords={CONSTITUENT_DIM: _names(constituents)},
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


def write_harmonic_constants(
    constants, store, chunk_cells=2**16, dtype="float32", *, chunk_constituents=-1
):
    """Write harmonic constants to Zarr with configurable cell and constituent chunks.

    Parameters
    ----------
    constants : xarray.Dataset
        Output of :func:`harmonic_constants_dataset` or :func:`constants_to_healpix`.
    store : str or path
    chunk_cells : int
        Cells per chunk (contiguous nested ranges when the cells are sorted).
    dtype : str
        Storage type of ``tide_real`` and ``tide_imag`` (float32: < 1e-7 m error per metre).
    chunk_constituents : int, optional
        Constituents per chunk. The default (-1) groups all constituents together.
    """
    from pangeo_fish.storage import _with_dggs_attrs

    _grid_info(constants["cell_ids"])
    order = np.argsort(constants["cell_ids"].values, kind="stable")
    ds = constants.isel({CELL_DIM: order})
    ds = ds.assign(
        {name: ds[name].astype(dtype) for name in ("tide_real", "tide_imag")}
    )
    ds = ds.assign_coords({CONSTITUENT_DIM: _names(ds[CONSTITUENT_DIM].values)})
    ds = _with_dggs_attrs(ds).chunk(
        {CONSTITUENT_DIM: chunk_constituents, CELL_DIM: chunk_cells}
    )
    for var in ds.variables.values():
        for key in (
            "chunks",
            "preferred_chunks",
            "compressors",
            "compressor",
            "shards",
        ):
            var.encoding.pop(key, None)
    ds.to_zarr(store, mode="w", zarr_format=2, consolidated=True)
    return store


def write_tide_pdf_healpix(likelihood, target_root, behav_found, step=200):
    """Write ``tide_pdf_healpix.zarr`` in the layout read by ``merge_tide_with_total``.

    Parameters
    ----------
    likelihood : xarray.Dataset
        Output of :func:`still_tag_tide_likelihood` with a daily ``time_step``.
    target_root : str or Path
        Folder receiving ``tide_pdf_healpix.zarr``.
    behav_found : array-like, pandas.Series or xarray.DataArray
        Daily behaviour flag on the time steps of ``likelihood`` (read downstream as
        the behaviour mask of the multi-sigma fit, e.g. ``behav_found > 0``). It is
        not derived from the tide likelihood, so it must be given explicitly.
        Time-labelled flags are aligned to the likelihood dates.

    Notes
    -----
    ``tide_found`` is written as 0/1 per window (the window is or is not used as
    evidence), a special case of the daily fraction written by ``tide_to_healpix``.
    """
    import zarr

    from pangeo_fish.storage import _with_dggs_attrs, write_time_blocks

    ds = likelihood[["pdf", "ocean_mask", "tide_found"]].copy()
    if isinstance(behav_found, xr.DataArray) and "time" in behav_found.coords:
        if behav_found.dims != ("time",):
            raise ValueError("behav_found must have only the time dimension")
        behav_found = behav_found.to_series()
    if isinstance(behav_found, pd.Series) and isinstance(
        behav_found.index, pd.DatetimeIndex
    ):
        flags_by_time = _as_series(behav_found)
        times = ds.time.to_index()
        if (
            not flags_by_time.index.is_unique
            or not times.isin(flags_by_time.index).all()
        ):
            raise ValueError(
                "behav_found must have a unique flag for each likelihood time"
            )
        behav_found = flags_by_time.reindex(times)
    flags = np.asarray(getattr(behav_found, "values", behav_found), dtype=float)
    if flags.shape != (ds.sizes["time"],):
        raise ValueError(
            f"behav_found has shape {flags.shape}, expected ({ds.sizes['time']},)"
        )
    ds["behav_found"] = ("time", flags)
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
            CONSTITUENT_DIM: _names(constituents),
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
    if lon_min > lon_max:
        raise ValueError(
            "bbox longitude must be increasing in -180..180; a box across the"
            " antimeridian is not supported"
        )
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
        missing = [c for c in constituents if str(c).lower() not in lower]
        if missing:
            raise KeyError(f"constituents not in the TMD file {names}: {missing}")
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


def fes2022b_constants_to_healpix(source, cell_ids, constituents=None):
    """Interpolate FES2022b native LGP2 harmonic constants to WGS84 HEALPix cells.

    Parameters
    ----------
    source : str or pathlib.Path
        Local AVISO ``FES2022b_OceanTide_NSgrid.nc`` file.
    cell_ids : xarray.DataArray
        Target nested HEALPix cells with WGS84 grid attributes.
    constituents : sequence of str, optional
        Constituents to interpolate, in the requested order. The default uses
        all constituents in the file.

    Returns
    -------
    xarray.Dataset
        Complex constants in metres, stored as real and imaginary parts.
        Cells outside the native mesh have NaN constants.
    """
    if str(_grid_info(cell_ids)["ellipsoid"]).casefold() != "wgs84":
        raise ValueError("FES2022b interpolation requires WGS84 HEALPix cells")
    lon, lat = cell_centers(cell_ids)
    if not lon.size:
        raise ValueError("FES2022b interpolation requires at least one HEALPix cell")
    pyfes = _require_pyfes()

    with xr.open_dataset(source, decode_cf=False) as ds:
        required = {"lon", "lat", "triangle", "lgp2"}
        if missing := required.difference(ds.variables):
            raise ValueError(f"FES2022b native mesh is missing {sorted(missing)}")
        available = {
            name.removesuffix("_amplitude").casefold(): name.removesuffix("_amplitude")
            for name in ds.data_vars
            if name.endswith("_amplitude")
        }
        wanted = list(available) if constituents is None else list(constituents)
        if not wanted:
            raise ValueError("select at least one FES2022b constituent")
        if missing := [name for name in wanted if name.casefold() not in available]:
            raise ValueError(f"constituents missing from FES2022b: {missing}")
        names = [available[name.casefold()] for name in wanted]
        for name in names:
            if f"{name}_phase" not in ds:
                raise ValueError(f"FES2022b native file is missing {name}_phase")
            if ds[f"{name}_amplitude"].attrs.get("units") != "cm":
                raise ValueError(f"FES2022b amplitude for {name} must be in cm")
        attrs = {
            key: ds.attrs[key]
            for key in (
                "license",
                "credits",
                "doi",
                "product_version",
                "date_created",
                "references",
            )
            if key in ds.attrs
        }

    # Use the shorter longitude interval for regional reads near the antimeridian.
    shifted = lon % 360
    bounds = shifted if np.ptp(shifted) < np.ptp(lon) else lon
    west = (bounds.min() - 0.5 + 180) % 360 - 180
    east = (bounds.max() + 0.5 + 180) % 360 - 180
    bbox = None
    if np.ptp(bounds) < 359:
        bbox = (west, max(-90, lat.min() - 0.5), east, min(90, lat.max() + 0.5))

    model = pyfes.config.LGP(
        path=str(source),
        type="lgp2",
        codes="lgp2",
        tidal_type="tide",
        constituents=names,
        bbox=bbox,
    ).load()
    waves, quality = model.interpolate(lon, lat)
    by_name = {key.casefold(): value for key, value in waves.items()}
    constants = np.stack([by_name[name.casefold()] for name in names]) * 0.01
    constants[:, quality <= 0] = np.nan + 1j * np.nan
    attrs.update(
        source="FES2022b Ocean Tide (native mesh)",
        tide_model="fes2022b",
        source_product="FES2022b_OceanTide_NSgrid.nc",
        source_url="https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html",
        distributor="AVISO+",
        creators="LEGOS, NOVELTIS, CLS",
        licensor="CNES",
        resampling="PyFES LGP2 interpolation of complex harmonic constants",
    )
    return harmonic_constants_dataset(constants, names, cell_ids, attrs=attrs)
