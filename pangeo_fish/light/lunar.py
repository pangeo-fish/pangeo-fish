"""Lunar template-fit geolocation method (Part B of the light pipeline).

Identifies usable moon nights, computes Spearman correlation between
the observed light curve and the physically predicted moonlight curve at
each grid pixel, converts correlations to likelihoods, and stacks the
results into an xarray DataArray ready for HEALPix regridding.
"""

import time as tmod

import ephem
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

from pangeo_fish.light.physics import predicted_moon_curve


def find_usable_moon_nights(
    df,
    min_phase=60,
    max_depth_m=15,
    bin_min=15,
    min_bins=8,
    min_samples_per_bin=3,
):
    """Identify calendar nights that are usable for lunar geolocation.

    A night is usable if:

    * Mean moon phase > ``min_phase`` (bright enough).
    * At least 30 shallow observations (``pressure < max_depth_m``).
    * At least ``min_bins`` time bins of ``bin_min`` minutes, each with
      ≥ ``min_samples_per_bin`` samples.

    Parameters
    ----------
    df : pd.DataFrame
        Tag data with columns ``time`` (UTC), ``light``, ``pressure``.
    min_phase : float, default 60
        Minimum moon illuminated fraction (%) at midnight.
    max_depth_m : float, default 15
        Maximum depth (m / dbar) considered "shallow".
    bin_min : int, default 15
        Duration of each time bin (minutes).
    min_bins : int, default 8
        Minimum number of valid bins required per night.
    min_samples_per_bin : int, default 3
        Minimum number of raw samples per bin.

    Returns
    -------
    list of dict
        Each dict has keys:
        ``date`` (pd.Timestamp), ``phase`` (float, %),
        ``times`` (np.ndarray of bin-centre datetimes),
        ``light`` (np.ndarray of binned median light counts),
        ``n_bins`` (int), ``n_raw`` (int).
    """
    all_dates = pd.date_range(
        df["time"].min().normalize(),
        df["time"].max().normalize(),
        freq="D",
        tz="UTC",
    )
    moon_nights = []

    for d in all_dates:
        # Check moon phase at ~01:00 UTC (representative of the night)
        obs_chk = ephem.Observer()
        obs_chk.pressure = 0
        obs_chk.date = (d + pd.Timedelta(hours=1)).strftime("%Y/%m/%d %H:%M:%S")
        phase = float(ephem.Moon(obs_chk).phase)

        if phase < min_phase:
            continue

        # Night window: 19:00 UTC on day d to 08:00 UTC on day d+1 (32 h span)
        t0 = d + pd.Timedelta(hours=19)
        t1 = d + pd.Timedelta(hours=32)

        mask = (df["time"] >= t0) & (df["time"] <= t1) & (df["pressure"] < max_depth_m)
        sub = df[mask].copy()
        if len(sub) < 30:
            continue

        # Bin into ``bin_min``-minute medians
        sub["bin"] = sub["time"].dt.floor(f"{bin_min}min")
        binned = (
            sub.groupby("bin")
            .agg(light_med=("light", "median"), n=("light", "count"))
            .reset_index()
        )
        binned = binned[binned["n"] >= min_samples_per_bin]

        if len(binned) < min_bins:
            continue

        moon_nights.append(
            {
                "date": d,
                "phase": phase,
                "times": binned["bin"].dt.tz_localize(None).values,
                "light": binned["light_med"].values,
                "n_bins": len(binned),
                "n_raw": len(sub),
            }
        )

    return moon_nights


def compute_lunar_correlation_maps(moon_nights, lons_moon, lats_moon):
    """Compute Spearman correlation maps for each usable moon night.

    For each night and each ``(lat, lon)`` pixel, computes the Spearman
    correlation between the binned observed light curve and the physically
    predicted moonlight curve via
    :func:`~pangeo_fish.light.physics.predicted_moon_curve`.

    The function augments each dict in ``moon_nights`` **in place** with:

    * ``corr_map`` — ``np.ndarray`` of shape ``(n_lats, n_lons)``
    * ``best_lon``, ``best_lat``, ``best_r`` — peak correlation location
    * ``c_lon``, ``c_lat`` — positive-correlation-weighted centroid
    * ``contrast`` — ``nanmax - nanmin`` of the correlation map

    Parameters
    ----------
    moon_nights : list of dict
        As returned by :func:`find_usable_moon_nights`.
    lons_moon : array-like
        1-D longitude axis (degrees East).
    lats_moon : array-like
        1-D latitude axis (degrees North).

    Returns
    -------
    list of dict
        Same list (mutated in place), returned for convenience.

    Notes
    -----
    This is computationally intensive (~minutes for a year of data on a
    1° grid).  Progress is printed every 10 nights.  A minimum of 6 valid
    time points are required for the Spearman test; pixels with fewer are
    left as ``NaN``.
    """
    lons_moon = np.asarray(lons_moon)
    lats_moon = np.asarray(lats_moon)

    for ni, mn in enumerate(moon_nights):
        t0_comp = tmod.time()
        times_bin = mn["times"]
        light_bin = mn["light"]

        corr_map = np.full((len(lats_moon), len(lons_moon)), np.nan)

        for i, lat in enumerate(lats_moon):
            for j, lon in enumerate(lons_moon):
                pred = predicted_moon_curve(times_bin, lon, lat)
                valid = ~np.isnan(pred) & (pred > 0)
                if valid.sum() >= 6:
                    r, _ = spearmanr(light_bin[valid], pred[valid])
                    corr_map[i, j] = r

        mn["corr_map"] = corr_map

        if not np.all(np.isnan(corr_map)):
            best = np.unravel_index(np.nanargmax(corr_map), corr_map.shape)
            mn["best_lon"] = float(lons_moon[best[1]])
            mn["best_lat"] = float(lats_moon[best[0]])
            mn["best_r"] = float(np.nanmax(corr_map))

            # Positive-correlation-weighted centroid
            pc = np.clip(corr_map, 0, None)
            pc[np.isnan(pc)] = 0.0
            tot = pc.sum()
            lg, la = np.meshgrid(lons_moon, lats_moon)
            mn["c_lon"] = float(np.sum(pc / tot * lg)) if tot > 0 else np.nan
            mn["c_lat"] = float(np.sum(pc / tot * la)) if tot > 0 else np.nan
            mn["contrast"] = float(np.nanmax(corr_map) - np.nanmin(corr_map))
        else:
            mn["best_lon"] = mn["best_lat"] = mn["best_r"] = np.nan
            mn["c_lon"] = mn["c_lat"] = mn["contrast"] = np.nan

        dt = tmod.time() - t0_comp
        if ni % 10 == 0 or ni == len(moon_nights) - 1:
            print(
                f"  [{ni + 1:3d}/{len(moon_nights)}] {mn['date'].date()} "
                f"ph={mn['phase']:3.0f}% r={mn.get('best_r', 0):.3f} "
                f"c=({mn.get('c_lon', 0):+.1f}°, {mn.get('c_lat', 0):.1f}°) "
                f"[{dt:.1f}s]"
            )

    return moon_nights


def correlation_to_likelihood(moon_nights, sigma_r=0.25):
    """Transform Spearman correlation maps into proper likelihood maps.

    Raw Spearman ``r ∈ [-1, 1]`` is not a likelihood — negative values
    have no probabilistic meaning.  The transformation:

        L_moon(λ, φ) = exp((r(λ, φ) − r_min) / σ_r)

    maps the full range of ``r`` to positive values, then normalises so
    the maximum is 1.

    Parameters
    ----------
    moon_nights : list of dict
        Must have ``corr_map`` set (output of
        :func:`compute_lunar_correlation_maps`).
    sigma_r : float, default 0.25
        Sharpness parameter.  Larger values → smoother likelihood.
        Default (0.25) is tuned on Mediterranean bluefin tuna data;
        an earlier version used 0.15.

    Returns
    -------
    list of dict
        Same list (mutated in place); each dict gains ``lh_map``
        (``np.ndarray``, same shape as ``corr_map``, max = 1,
        NaN where ``corr_map`` was NaN).

    Notes
    -----
    Do **not** use raw Spearman ``r`` as a likelihood directly — the
    anti-checklist in the spec is explicit on this point.
    """
    for mn in moon_nights:
        cm = mn.get("corr_map")
        if cm is None or np.all(np.isnan(cm)):
            mn["lh_map"] = None
            continue
        r_min = float(np.nanmin(cm))
        lh = np.exp((cm - r_min) / sigma_r)
        lh[np.isnan(cm)] = np.nan
        lh /= np.nanmax(lh)  # normalize to [0, 1], max = 1
        mn["lh_map"] = lh
    return moon_nights


def lunar_likelihood_to_dataarray(moon_nights, lons_moon, lats_moon):
    """Stack per-night likelihood maps into an xarray DataArray.

    Parameters
    ----------
    moon_nights : list of dict
        Must have ``lh_map`` set (output of
        :func:`correlation_to_likelihood`).
    lons_moon : array-like
        1-D longitude axis (degrees East).
    lats_moon : array-like
        1-D latitude axis (degrees North).

    Returns
    -------
    xr.DataArray
        Dims ``('time', 'latitude', 'longitude')``.  ``time`` is the
        night date as UTC ``datetime64[ns]``.  Nights without a valid
        ``lh_map`` are filled with ``NaN`` — the merge step in the
        notebook treats missing nights as a uniform (uninformative) prior.
        Attributes: ``method='lunar_template_fit'``, ``sigma_r`` is
        recorded if available in the first night's metadata.

    Notes
    -----
    The DataArray is named ``'pdf_moon'`` to be compatible with
    :func:`~pangeo_fish.pdf.combine_emission_pdf_with_light`.
    """
    lons_moon = np.asarray(lons_moon)
    lats_moon = np.asarray(lats_moon)
    shape = (len(lats_moon), len(lons_moon))

    times = []
    maps = []
    for mn in moon_nights:
        times.append(pd.Timestamp(mn["date"].date()))
        lh = mn.get("lh_map")
        if lh is None:
            maps.append(np.full(shape, np.nan))
        else:
            maps.append(np.where(np.isnan(lh), np.nan, lh))

    da = xr.DataArray(
        np.stack(maps, axis=0),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "latitude": lats_moon,
            "longitude": lons_moon,
        },
        name="pdf_moon",
        attrs={"method": "lunar_template_fit"},
    )
    return da


def apply_cloud_weighting(moon_nights, era5_ds, base_sigma=0.5, cloud_penalty=0.5):
    """Apply ERA5 cloud-cover weighting to lunar likelihood maps.

    .. todo::
        This function is a stub — ERA5 cloud weighting is deferred to a
        second iteration.  Enable it by calling after
        :func:`correlation_to_likelihood` and before
        :func:`lunar_likelihood_to_dataarray`.  Requires ``cdsapi``
        credentials and ERA5 cloud-cover data (see reference notebook
        cells 61–66 for the full implementation).

        To enable in the notebook::

            # fetch era5_ds via cdsapi for the deployment period
            moon_nights = apply_cloud_weighting(
                moon_nights, era5_ds,
                base_sigma=0.5, cloud_penalty=0.5,
            )

    Parameters
    ----------
    moon_nights : list of dict
        With ``lh_map`` set.
    era5_ds : xr.Dataset
        ERA5 total cloud cover dataset (variable ``tcc``, 0–1).
    base_sigma : float, default 0.5
        Base sigma for cloud-modulated likelihood smoothing.
    cloud_penalty : float, default 0.5
        Scaling factor applied to cloudy pixels.

    Returns
    -------
    list of dict
        Unchanged (stub — no modification applied).
    """
    # TODO: implement ERA5 cloud weighting (reference notebook cells 61–66).
    # This requires cdsapi credentials and is not part of the default pipeline.
    raise NotImplementedError(
        "ERA5 cloud weighting is deferred — see the TODO in this function's "
        "docstring and reference notebook cells 61–66."
    )
