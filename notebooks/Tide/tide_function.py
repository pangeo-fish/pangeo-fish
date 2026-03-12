import xarray as xr
import pandas as pd
from ipywidgets import interact, IntSlider
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from numpy.linalg import lstsq
from tqdm import tqdm
from scipy.signal import savgol_filter
import xdggs
import sparse
from tqdm import tqdm
from scipy.stats import multivariate_normal


#####################################################################
#  Première partie: Extraction signal de marée par sinus
#####################################################################


def lssinfit(time, depth):
    """
    Ajuste un modèle sinusoïdal simple sur la profondeur.
    Retourne rmse, rsquare, amplitude.
    """

    depth_smoothed = savgol_filter(depth, window_length=15, polyorder=2, mode="mirror")

    w = 2 * np.pi / (12.42 / 24)  # fréquence de marée (rad/jour)
    X = np.column_stack([np.ones_like(time), np.sin(w * time), np.cos(w * time)])
    # print(depth_smoothed, X)
    # print(len(time),len(depth_smoothed))
    coef, _, _, _ = lstsq(X, depth_smoothed, rcond=None)
    fitted = X @ coef
    residuals = depth_smoothed - fitted

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((depth - np.mean(depth)) ** 2)
    rsq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(residuals**2))
    amplitude = np.sqrt(coef[1] ** 2 + coef[2] ** 2)
    return rmse, rsq, amplitude  # , depth_smoothed


def tide_behav_extr(
    df,
    tagno,
    tideFL=10,
    tideLV=[0.5, 0.7, 0.5],
    behavFL=16,
    behavLV=[0.5, 0.7, 0.5],
    dt=10,
):
    """
    Traduction Python de la fonction MATLAB tidebehavextr.m
    Entrée : CSV avec colonnes 'time', 'depth' (et éventuellement 'temp')
    Sortie : xarray.Dataset avec résultats de classification marée / comportement
    """
    # --- Lecture des données brutes ---

    if "time" not in df.columns or "pressure" not in df.columns:
        raise ValueError("Le CSV doit contenir au moins 'time' et 'pressure'")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    time = (
        (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 3600.0 / 24.0
    )  # en jours
    depth = df["pressure"].to_numpy()
    print(df["time"] - df["time"].iloc[0])
    n = len(depth)
    tide_window = int(round(60 / dt * tideFL))
    print(tide_window)
    behav_window = int(round(60 / dt * behavFL))

    # --- Initialisation ---
    rmse_tide = np.full(n, np.nan)
    rsq_tide = np.full(n, np.nan)
    ampli_tide = np.full(n, np.nan)
    # depth_smoothed_list = [[] for _ in range(n)]

    rmse_behav = np.full(n, np.nan)
    rsq_behav = np.full(n, np.nan)
    ampli_behav = np.full(n, np.nan)

    # --- Boucle glissante pour marée ---
    print(f"Extraction du signal de marée ({tideFL} h)...")
    for i in tqdm(range(0, n - tide_window)):
        idx = slice(i, i + tide_window)
        rmse, rsq, ampli = lssinfit(time[idx], depth[idx])
        rmse_tide[i] = rmse
        rsq_tide[i] = rsq
        ampli_tide[i] = ampli
        # depth_smoothed_list[i] = depth_lissage

    # --- Détection des intervalles de marée ---
    tide_found = (
        (rmse_tide < tideLV[0]) & (rsq_tide > tideLV[1]) & (ampli_tide > tideLV[2])
    )

    # --- Boucle glissante pour comportement ---
    print(f"Classification comportementale ({behavFL} h)...")
    for i in tqdm(range(0, n - behav_window)):
        idx = slice(i, i + behav_window)
        rmse, rsq, ampli = lssinfit(time[idx], depth[idx])
        rmse_behav[i] = rmse
        rsq_behav[i] = rsq
        ampli_behav[i] = ampli

    behav_found = (
        (rmse_behav < behavLV[0])
        & (rsq_behav > behavLV[1])
        & (ampli_behav > behavLV[2])
    )

    # --- Création du dataset de sortie ---
    ds = xr.Dataset(
        data_vars=dict(
            depth=("time", depth),
            rmse_tide=("time", rmse_tide),
            rsq_tide=("time", rsq_tide),
            ampli_tide=("time", ampli_tide),
            tide_found=("time", tide_found.astype(int)),
            rmse_behav=("time", rmse_behav),
            rsq_behav=("time", rsq_behav),
            ampli_behav=("time", ampli_behav),
            behav_found=("time", behav_found.astype(int)),
        ),
        coords=dict(
            time=("time", df["time"]),
        ),
        attrs=dict(
            tagno=str(tagno),
            tideFL=tideFL,
            tideLV=tideLV,
            behavFL=behavFL,
            behavLV=behavLV,
            dt=dt,
        ),
    )

    print("\nExtraction terminée ✅")
    return ds


#####################################################################
# Traitement du dataset de Moko
#####################################################################


def convert_lon_360_to_180(ds, bbox=None):
    """
    Convertit les longitudes d'un Dataset de 0-360 à -180-180,
    applique une bbox, et utilise nearest si les bornes
    exactes ne sont pas dans les coordonnées.

    Parameters
    ----------
    ds : xarray.Dataset

    bbox : dict or None
        Exemple :
        bbox = {"latitude": [lat_min, lat_max], "longitude": [lon_min, lon_max]}
    """

    ds = ds.copy()

    # --- 1. Conversion lon 0–360 → –180–180 ---
    lon_new = ((ds.lon + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon_new)
    ds = ds.sortby("lon")

    # --- 2. Application bbox ---
    if bbox is not None:
        # -------- LATITUDE --------
        if "latitude" in bbox:
            lat_min, lat_max = bbox["latitude"]
            print(lat_min, lat_max)
            # Vérifie si les valeurs existent
            lat_vals = ds.lat.values
            # min
            if lat_min in lat_vals:
                print("here")
                lat_min_sel = lat_min
            else:
                lat_min_sel = ds.lat.sel(lat=lat_min, method="nearest").item()
            # max
            if lat_max in lat_vals:
                print("HERE")
                lat_max_sel = lat_max
            else:
                lat_max_sel = ds.lat.sel(lat=lat_max, method="nearest").item()
                print(lat_max_sel)
            print(slice(lat_min_sel, lat_max_sel))
            ds = ds.sortby("lat")  # latitudes croissantes
            ds = ds.sel(lat=slice(lat_min_sel, lat_max_sel))
        # -------- LONGITUDE --------
        if "longitude" in bbox:
            lon_min, lon_max = bbox["longitude"]
            lon_vals = ds.lon.values
            # min
            if lon_min in lon_vals:
                lon_min_sel = lon_min
            else:
                lon_min_sel = ds.lon.sel(lon=lon_min, method="nearest").item()
            # max
            if lon_max in lon_vals:
                lon_max_sel = lon_max
            else:
                lon_max_sel = ds.lon.sel(lon=lon_max, method="nearest").item()
            ds = ds.sel(lon=slice(lon_min_sel, lon_max_sel))

    return ds


#####################################################################


def datalikelihood_fast_xr_multi_vec_mask(td_depth, td_time, tidal_ds, sigma_tid=2.0):
    """
    Version vectorisée du datalikelihood (mode fast) avec masque océan/terre.

    td_depth : np.array, shape (n_times, n_meas)
    td_time  : np.array, shape (n_times,)
    tidal_ds : xarray.Dataset
    sigma_tid: float, écart-type de l'erreur
    """
    lat = tidal_ds.lat.values
    lon = tidal_ds.lon.values
    mask = tidal_ds.mask.values.astype(bool)  # True = océan, False = terre
    constituents = tidal_ds.constituents.values
    n_times, n_meas = td_depth.shape
    n_lat, n_lon = len(lat), len(lon)

    # Variables du dataset
    amp = tidal_ds.amplitude.values  # (constituents, lon)
    phase = tidal_ds.phase.values  # (constituents, lon)
    omega = tidal_ds.omega.values  # (constituents, lon)
    depth_mean = tidal_ds.wct.values  # (lat, lon)

    # Calcul des marées prédites
    td_time_exp = td_time[:, None, None]  # (n_times, 1, 1)
    amp_exp = amp[None, :, :]
    phase_exp = phase[None, :, :]
    omega_exp = omega[None, :, :]

    cos_term = np.cos(omega_exp * td_time_exp + phase_exp)
    tidal_pred = amp_exp * cos_term  # (n_times, n_const, n_lon)
    tidal_sum = tidal_pred.sum(axis=1)  # (n_times, n_lon)
    mu = -tidal_sum[:, None, :] + depth_mean[None, :, :]  # (n_times, n_lat, n_lon)

    # LIK[t, lat, lon] : vecteur pour temps seulement
    LIK = np.zeros((n_times, n_lat, n_lon))
    for t in tqdm(range(n_times), desc="Temps"):
        td_mean = np.nanmean(td_depth[t, :])
        mu_eff = mu[t, :, :]
        # Calcul de la vraisemblance
        LIK[t, :, :] = np.exp(-0.5 * ((td_mean - mu_eff) / sigma_tid) ** 2) / (
            sigma_tid * np.sqrt(2 * np.pi)
        )
        # Appliquer le masque : vraisemblance sur terre = NaN
        LIK[t, ~mask] = np.nan

    return LIK


def datalikelihood_full_xr_vec(td_depth, td_time, tidal_ds, sigma_tid=2.0):
    """
    Version vectorisée 'full' du datalikelihood avec application du mask.

    td_depth : np.array, shape (n_times, n_meas)
    td_time  : np.array, shape (n_times,)
    tidal_ds : xarray.Dataset (doit contenir 'mask')
    sigma_tid: float, si pas de covariance, utilisé comme diag
    """
    lat = tidal_ds.lat.values
    lon = tidal_ds.lon.values
    constituents = tidal_ds.constituents.values
    n_times, n_meas = td_depth.shape
    n_lat, n_lon = len(lat), len(lon)

    # Pré-calcul des marées
    amp = tidal_ds.amplitude.values  # (n_const, n_lon)
    phase = tidal_ds.phase.values
    omega = tidal_ds.omega.values
    depth_mean = tidal_ds.wct.values  # (n_lat, n_lon)

    # Mask (1=water, 0=land)
    # mask = tidal_ds.mask.values            # shape (n_lat, n_lon)
    mask = tidal_ds.mask.values.astype(bool)  # True = océan, False = terre
    td_time_exp = td_time[:, None, None]  # (n_times,1,1)
    amp_exp = amp[None, :, :]  # (1, n_const, n_lon)
    phase_exp = phase[None, :, :]
    omega_exp = omega[None, :, :]

    tidal_pred = amp_exp * np.cos(
        omega_exp * td_time_exp + phase_exp
    )  # (n_times, n_const, n_lon)
    tidal_sum = tidal_pred.sum(axis=1)  # (n_times, n_lon)
    mu = -tidal_sum[:, None, :] + depth_mean[None, :, :]  # (n_times, n_lat, n_lon)

    LIK = np.zeros((n_times, n_lat, n_lon))

    for t in tqdm(range(n_times), desc="Temps"):
        x = td_depth[t, :]  # shape (n_meas,)

        # Covariance empirique ou fallback si n_meas=1
        if n_meas == 1:
            cov = np.array([[sigma_tid**2]])
        else:
            cov = np.cov(x, rowvar=False)
            if cov.shape == ():  # cas particulier
                cov = np.array([[cov]])
            else:
                cov = np.atleast_2d(cov)

        # Moyenne des écarts-types pour approximation diagonal
        sigma_vec = np.sqrt(np.diag(cov)).mean()
        if sigma_vec < 1e-6:  # seuil arbitraire
            sigma_vec = sigma_tid

        # LIK full approx
        mu_ij = mu[t, :, :]  # shape (n_lat, n_lon)
        x_mean = np.nanmean(td_depth[t, :])
        LIK_t = np.exp(-0.5 * ((x_mean - mu_ij) / sigma_vec) ** 2) / (
            sigma_vec * np.sqrt(2 * np.pi)
        )

        # Appliquer le mask : 0 sur la terre, garder la valeur sur l'eau
        # LIK[t, :, :] = LIK_t * mask
        LIK_t_masked = np.where(mask, LIK_t, np.nan)
        LIK[t, :, :] = LIK_t_masked
    return LIK


#####################################################################################################
###   PDF que sur les instants de marée   ###
#####################################################################################################

# Méthode Fast avec le sigma_tid


def datalikelihood_tide_only_normalized(
    td_depth, td_time, tidal_ds, tide_found, sigma_tid=2.0
):
    """
    Calcule une PDF normalisée pour chaque instant.
    - Si marée détectée : PDF = vraisemblance (gaussienne)
    - Sinon : PDF uniforme sur l'océan (somme = 1)

    Retourne un Dataset avec une PDF normalisée pour chaque (t,lat,lon).
    """
    eps = 1e-200
    lat = tidal_ds.lat.values
    lon = tidal_ds.lon.values
    mask = tidal_ds.mask.values.astype(bool)  # True = océan

    n_times, n_meas = td_depth.shape
    n_lat, n_lon = len(lat), len(lon)

    # Marée pré-calculée
    amp = tidal_ds.amplitude.values
    phase = tidal_ds.phase.values
    omega = tidal_ds.omega.values
    depth_mean = tidal_ds.wct.values

    td_time_exp = td_time[:, None, None]
    amp_exp = amp[None, :, :]
    phase_exp = phase[None, :, :]
    omega_exp = omega[None, :, :]

    tidal_pred = amp_exp * np.cos(omega_exp * td_time_exp + phase_exp)
    tidal_sum = tidal_pred.sum(axis=1)
    mu = -tidal_sum[:, None, :] + depth_mean[None, :, :]  # (n_times, n_lat, n_lon)

    # PDF finale
    PDF = np.full((n_times, n_lat, n_lon), np.nan)

    tide_idx = np.where(tide_found == 1)[0]
    notide_idx = np.where(tide_found == 0)[0]

    # --- Cas 1 : marée détectée → PDF gaussienne normalisée ---
    for t in tqdm(tide_idx, desc="PDF avec marée détectée"):
        x_mean = np.nanmean(td_depth[t, :])
        mu_eff = mu[t, :, :]

        L = np.exp(-0.5 * ((x_mean - mu_eff) / sigma_tid) ** 2)
        L = L * mask  # appliquer masque

        # normalisation pour somme=1
        s = np.nansum(L)

        if s == 0:
            PDF[t, :, :] = mask / np.sum(mask)  # fallback uniforme
        else:
            PDF[t, :, :] = L / s
        PDF[t, :, :] = np.maximum(PDF[t, :, :], eps)
        PDF[t, :, :] /= np.nansum(PDF[t, :, :])
    # --- Cas 2 : pas de marée détectée → PDF uniforme ---
    ocean_pixels = np.sum(mask)
    print("ocean_pixel", ocean_pixels)
    uniform_pdf = mask / ocean_pixels
    uniform_pdf = np.maximum(uniform_pdf, eps)
    uniform_pdf /= np.nansum(uniform_pdf)
    print("somme pdf", uniform_pdf.sum())
    print("pas de tide nombre d'indice", notide_idx)

    for t in notide_idx:
        PDF[t, :, :] = uniform_pdf

    # Construction Dataset
    ds_pdf = xr.Dataset(
        {
            "pdf": (("time", "latitude", "longitude"), PDF),
            "ocean_mask": (("latitude", "longitude"), mask),
            "tide_found": (("time"), tide_found),
        },
        coords={"time": td_time, "latitude": lat, "longitude": lon},
    )

    return ds_pdf


###_______________________________________________________________________________________________________###
### METHOD TIDE FULL NORALIZED ###
###_______________________________________________________________________________________________________###


def datalikelihood_tide_only_full_normalized(
    td_depth, td_time, tidal_ds, tide_found, sigma_tid=2.0
):
    """
    Calcule une PDF normalisée pour chaque instant.
    - Si marée détectée : PDF = vraisemblance (gaussienne)
    - Sinon : PDF uniforme sur l'océan (somme = 1)

    Retourne un Dataset avec une PDF normalisée pour chaque (t,lat,lon).
    """
    eps = 1e-200
    lat = tidal_ds.lat.values
    lon = tidal_ds.lon.values
    mask = tidal_ds.mask.values.astype(bool)  # True = océan

    n_times, n_meas = td_depth.shape
    n_lat, n_lon = len(lat), len(lon)

    # Marée pré-calculée
    amp = tidal_ds.amplitude.values
    phase = tidal_ds.phase.values
    omega = tidal_ds.omega.values
    depth_mean = tidal_ds.wct.values

    td_time_exp = td_time[:, None, None]
    amp_exp = amp[None, :, :]
    phase_exp = phase[None, :, :]
    omega_exp = omega[None, :, :]

    tidal_pred = amp_exp * np.cos(omega_exp * td_time_exp + phase_exp)
    tidal_sum = tidal_pred.sum(axis=1)
    mu = -tidal_sum[:, None, :] + depth_mean[None, :, :]  # (n_times, n_lat, n_lon)

    # PDF finale
    PDF = np.full((n_times, n_lat, n_lon), np.nan)

    tide_idx = np.where(tide_found == 1)[0]
    notide_idx = np.where(tide_found == 0)[0]

    # --- Cas 1 : marée détectée → PDF gaussienne normalisée ---
    for t in tqdm(tide_idx, desc="PDF avec marée détectée"):
        x = td_depth[t, :]  # shape (n_meas,)

        # Covariance empirique ou fallback si n_meas=1
        if n_meas == 1:
            cov = np.array([[sigma_tid**2]])
        else:
            cov = np.cov(x, rowvar=False)
            if cov.shape == ():  # cas particulier
                cov = np.array([[cov]])
            else:
                cov = np.atleast_2d(cov)

        # Moyenne des écarts-types pour approximation diagonal
        sigma_vec = np.sqrt(np.diag(cov)).mean()
        if sigma_vec < 1e-6:  # seuil arbitraire
            sigma_vec = sigma_tid

        # mu_eff = mu[t, :, :]
        ###
        mu_ij = mu[t, :, :]  # shape (n_lat, n_lon)
        x_mean = np.nanmean(td_depth[t, :])
        L = np.exp(-0.5 * ((x_mean - mu_ij) / sigma_vec) ** 2) / (
            sigma_vec * np.sqrt(2 * np.pi)
        )
        ###
        ###L = np.exp(-0.5 * ((x_mean - mu_eff)/sigma_tid)**2)
        L = L * mask  # appliquer masque

        # normalisation pour somme=1
        s = np.nansum(L)

        if s == 0:
            PDF[t, :, :] = mask / np.sum(mask)  # fallback uniforme
        else:
            PDF[t, :, :] = L / s
        PDF[t, :, :] = np.maximum(PDF[t, :, :], eps)
        PDF[t, :, :] /= np.nansum(PDF[t, :, :])
    # --- Cas 2 : pas de marée détectée → PDF uniforme ---
    ocean_pixels = np.sum(mask)

    uniform_pdf = mask / ocean_pixels
    uniform_pdf = np.maximum(uniform_pdf, eps)
    uniform_pdf /= np.nansum(uniform_pdf)

    for t in notide_idx:
        PDF[t, :, :] = uniform_pdf

    # Construction Dataset
    ds_pdf = xr.Dataset(
        {
            "pdf": (("time", "latitude", "longitude"), PDF),
            "ocean_mask": (("latitude", "longitude"), mask),
            "tide_found": (("time"), tide_found),
        },
        coords={"time": td_time, "latitude": lat, "longitude": lon},
    )

    return ds_pdf
