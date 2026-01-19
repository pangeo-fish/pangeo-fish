###########
#Usual libraries
###########
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import math
import netCDF4 as NC4
import os
import datetime
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
###########
#Signal processing functions
###########
from scipy.signal import detrend,welch 
import xarray as xr
###########
#Additionals functions
###########
from scipy.io import loadmat
from scipy.stats.distributions import chi2
import numpy.matlib
from IPython import display
from scipy.signal import fftconvolve, welch
import hvplot.xarray

def stations_proches(ds, lat, lon, delta_lat=0.2, delta_lon=0.2):
    lat_min, lat_max = lat - delta_lat, lat + delta_lat
    lon_min, lon_max = lon - delta_lon, lon + delta_lon

    mask_nodes = (
        (ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max) &
        (ds["longitude"] >= lon_min) & (ds["longitude"] <= lon_max)
    )
    index_nodes = ds.node.values[mask_nodes]  # Les vrais IDs

    nodes = ds.sel(node=index_nodes)          # Sélection par valeur, pas par position
    return nodes, index_nodes

def station_proche(ds, lat, lon):
    dlat = ds["latitude"] - lat
    dlon = ds["longitude"] - lon
    
    # distance approximative en degrés (valable localement)
    dist2 = dlat**2 + dlon**2

def conv_time(date_debut, date_evt, pas, step='seconds'):
    
    # Calcul de la différence de temps
    diff_t = date_evt - date_debut
    # Conversion en secondes (ou autre unité selon step)
    if step == 'seconds':
        diff_unit = diff_t.total_seconds()
    elif step == 'minutes':
        diff_unit = diff_t.total_seconds() / 60
    elif step == 'hours':
        diff_unit = diff_t.total_seconds() / 3600
    elif step == 'days':
        diff_unit = diff_t.total_seconds() / (3600*24)
    else:
        raise ValueError("Unité de temps non supportée. Choisir 'seconds', 'minutes', 'hours' ou 'days'")
    
    # Calcul de l'indice
    indice = int(diff_unit / pas)
    
    return indice
    
    idx = dist2.argmin()
    print(idx)
    return ds.isel(node=idx)

def dispNewtonTH(f,dep):
# This function inverts the linear dispersion relation (2*pi*f)^2=g*k*tanh(k*dep) to get 
# k from f and dep. 2 Arguments: f and dep are frequency in Hertz and depth in meters
# % Uses Newton method. Original code from T.H.C. Herbers
    eps=0.000001
    g=9.81
    sig=2*np.pi*f
    Y=dep*sig**2./g 

    X=np.sqrt(Y)
    I=1
    F=1
    while abs(np.amax(F)) > eps:
        H=np.tanh(X)
        F=Y-X*H
        FD=-H-X/(np.cosh(X)**2)
        X=X-F/FD

    dispNewtonTH=X/dep

    return dispNewtonTH

# --------------------------------------------------------
# filtre de Gabor 1D
# --------------------------------------------------------
def gabor_filter(f0, sigma, fs, duration=60):
    """
    Crée un filtre de Gabor 1D (temps, fenêtre finie).
    f0 : fréquence centrale (Hz)
    sigma : largeur gaussienne (s)
    fs : fréquence d'échantillonnage
    duration : durée totale du filtre (s)
    """
    t = np.arange(-duration/2, duration/2, 1/fs)
    g = np.exp(-t**2 / (2 * sigma**2)) * np.exp(1j*2*np.pi*f0*t)
    return t, g / np.linalg.norm(g)  # normalisation


def bin_over_f(ds,newf):
    """
    Interpolate spectrum over a new frequency axis

    Args:
        ds    (xarray): Spectrum
        newf  (array) : new frequency axis
    Return:
        newds (xarray): Spectrum with new frequency axis
    """
    newds = ds.copy()
    #newspec2D = ds.spec2D.interp(frequency=newf)
    newspec1D = ds.ef.interp(f=newf)
    #newds = newds.drop_vars(["spec2D"])
    newds = newds.drop_vars(["ef"])
    newds = newds.drop("f")
    if 'f' in newds.dims : newds = newds.drop_dims(["f"])

    newds = newds.assign_coords({"f": newf})

    #newds = newds.assign({"spec2D": newspec2D})
    newds = newds.assign({"ef": newspec1D  })

    return newds

# test détection des périodes de calmes

def detect_calm_periods(
    data,
    fs=0.5,                      # fréquence d’échantillonnage (Hz)
    var_window_minutes=5,        # taille de la fenêtre pour calculer la variance
    calm_var_threshold=5.0,      # seuil de variance pour considérer "calme"
    min_calm_duration_minutes=10 # durée minimale pour valider une période calme
    ):
    """
    Détecte toutes les périodes calmes dans le signal.

    Parameters
    ----------
    data : array-like
        Série temporelle (1D)
    fs : float
        Fréquence d'échantillonnage en Hz
    var_window_minutes : float
        Taille de la fenêtre (minutes) pour la variance glissante
    calm_var_threshold : float
        Variance en dessous de ce seuil => calme
    min_calm_duration_minutes : float
        Longueur minimale (minutes) d'une période calme

    Returns
    -------
    calm_periods : list of dict
        Chaque dict contient {"start_idx": int, "end_idx": int, "duration_s": float}
    """

    data = np.asarray(data, dtype=float)
    n = len(data)

    # Fenêtre pour variance
    window_points = int(var_window_minutes * 60 * fs)
    if window_points < 1:
        raise ValueError("La fenêtre de variance doit être au moins de 1 point")

    # Calcul variance glissante
    rolling_var = pd.Series(data).rolling(window=window_points, center=True).var().values
        # Masque des périodes calmes
    calm_mask = rolling_var <= calm_var_threshold

    # Durée minimale
    min_len = int(min_calm_duration_minutes * 60 * fs)

    calm_periods = []
    inside_calm = False
    start_idx = None

    for i, is_calm in enumerate(calm_mask):
        if is_calm and not inside_calm:
            # début d'une période calme
            inside_calm = True
            start_idx = i
        elif not is_calm and inside_calm:
            # fin d'une période calme
            end_idx = i - 1
            if end_idx - start_idx + 1 >= min_len:
                calm_periods.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "duration_s": (end_idx - start_idx + 1) / fs
                })
            inside_calm = False

    # Si le calme dure jusqu'à la fin
    if inside_calm:
        end_idx = n - 1
        if end_idx - start_idx + 1 >= min_len:
            calm_periods.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration_s": (end_idx - start_idx + 1) / fs
            })

    return calm_periods



#Fonction pour calculer la différence entre WW3 et l'élévation des vagues corrigée

def selection_WW3(time_enter, ef_eel, freq, ds_WW3):
    time_sel = pd.Timestamp(time_enter)

    # 1) Sélectionner le temps voulu
    ds_slice = ds_WW3.sel(time=time_sel, method="nearest")

    # 2) Interpoler sur la nouvelle grille de fréquences
    ds_interp = ds_slice.interp(f=freq)

    # 3) Calcul de la différence en restant dans xarray
    # ef_eel doit être un DataArray avec un seul axe 'f'
    if not isinstance(ef_eel, xr.DataArray):
        ef_eel = xr.DataArray(ef_eel, coords={"f": freq}, dims=["f"])

    diff = ds_interp["ef"] - ef_eel  # broadcasting automatique

    diff = np.abs(diff)
    # diff garde les coordonnées (f, node) du modèle WW3
    return diff

#Test chat GPT comparaison de spectre
def compute_spectral_similarity(ef_model_da, ef_ref_da,time, time_index=0, eps=1e-12):
    """
    ef_model_da : DataArray (time,f,node) or (f,node)  -- modèle WW3
    ef_ref_da   : DataArray (f,)                      -- référence
    Returns: xr.Dataset with metrics per node
    """
    # --- normaliser les noms / extraire (f,node) pour un seul time ---
    if "time" in ef_model_da.dims:
        model = ef_model_da.isel(time=time_index, method = "nearest")   # (f,node)
    else:
        model = ef_model_da  # already (f,node)
    # interp si nécessaire : s'assurer que les fréquences correspondent
    if not np.allclose(model['f'].values, ef_ref_da['f'].values):
        model = model.interp(f=ef_ref_da['f'].values)

    # numpy arrays
    A = model.values.astype(float)   # shape (n_f, n_node)
    r = ef_ref_da.values.astype(float)  # shape (n_f,)

    n_f, n_node = A.shape

    # mask nodes all-NaN
    allnan = np.all(np.isnan(A), axis=0)

    # --- Precompute reference stats ---
    r_mean = np.nanmean(r)
    r_demean = r - r_mean
    r_norm = np.sqrt(np.nansum(r**2)) + eps
    r_std = np.nanstd(r) + eps
      # --- Cosine similarity (and spectral angle) ---
    dot = np.nansum(A * r[:, None], axis=0)         # (n_node,)
    normA = np.sqrt(np.nansum(A**2, axis=0)) + eps  # (n_node,)
    cosine = dot / (normA * r_norm)                 # in (-1,1)
    cosine[allnan] = np.nan
    spectral_angle = np.arccos(np.clip(cosine, -1, 1))  # 0 = identical

    # --- Pearson correlation across f ---
    A_mean = np.nanmean(A, axis=0)                  # (n_node,)
    A_demean = A - A_mean[None, :]
    cov = np.nansum(A_demean * r_demean[:, None], axis=0)
    A_std = np.nanstd(A, axis=0) + eps
    pearson = cov / (r_std * A_std)
    pearson[allnan] = np.nan

    # --- Jensen-Shannon divergence (symmetrized, bounded) ---
    # build probability vectors (positive and normalized)
    # ensure positivity: shift small negative/zero to eps
    p = np.maximum(r, eps)
    p = p / np.sum(p)
    q = np.maximum(A, eps)            # (n_f, n_node)
    q_sum = np.nansum(q, axis=0)
    # avoid divide-by-zero
    q_sum[q_sum == 0] = np.nan
    q = q / q_sum[None, :]
    # compute M = 0.5(p+q)
    M = 0.5 * (p[:, None] + q)
    # KL(p||M) and KL(q||M)
    KL_pM = np.nansum(p[:, None] * np.log(p[:, None] / M), axis=0)   # shape (n_node,)
    KL_qM = np.nansum(q * np.log(q / M), axis=0)
    JS = 0.5 * (KL_pM + KL_qM)  # Jensen-Shannon divergence
        # numeric cleanup
    JS = np.where(np.isfinite(JS), JS, np.nan)
    JS[allnan] = np.nan
    # optional: symmetric metric = sqrt(JS)
    JS_sqrt = np.sqrt(JS)

    # --- Weighted RMSE (weight = reference normalized) ---
    w = p  # using p normalized (sum=1)
    diff = A - r[:, None]
    wrmse = np.sqrt(np.nansum((w[:, None] * (diff**2)), axis=0))
    wrmse[allnan] = np.nan

    # --- log-space Pearson correlation (log(ef+eps)) ---
    log_r = np.log(r + eps)
    log_A = np.log(np.maximum(A, eps))
    # demean
    lr_mean = np.nanmean(log_r)
    la_mean = np.nanmean(log_A, axis=0)
    num = np.nansum((log_A - la_mean[None,:]) * (log_r - lr_mean)[:, None], axis=0)
    den = (np.nanstd(log_A, axis=0) + eps) * (np.nanstd(log_r) + eps)
    log_pearson = num / den
    log_pearson[allnan] = np.nan
    # --- assemble xarray Dataset keyed by node ---
    coords = {"node": model.node.values,"time":time}
    ds = xr.Dataset({
        "cosine": ("node", cosine),
        "spectral_angle": ("node", spectral_angle),
        "pearson": ("node", pearson),
        "log_pearson": ("node", log_pearson),
        "JS": ("node", JS),
        "JS_sqrt": ("node", JS_sqrt),
        "wrmse": ("node", wrmse)
    }, coords=coords)

    return ds

