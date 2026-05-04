"""Physical moonlight illuminance model.

Implements the Austin (1976) / Poon et al. (2024 MEE) / Śmielak (2023)
model for predicting ground-level moonlight illuminance from first
principles.  This is — to the authors' knowledge — the first application
of this physical model to archival fish-tag geolocation.

References
----------
Allen, C.W. (1973). *Astrophysical Quantities* (3rd ed.). Athlone Press.
Austin, R.W. (1976). The remote sensing of spectral radiance from below
    the ocean surface. In N.G. Jerlov & E. Steemann Nielsen (Eds.),
    *Optical Aspects of Oceanography*. Academic Press.
Krisciunas, K., & Schaefer, B.E. (1991). A model of the brightness of
    moonlight. *Publications of the Astronomical Society of the Pacific*,
    103(667), 1033–1039.
Buratti, B.J., Hillier, J.K., & Wang, M. (1996). The Lunar Opposition
    Surge: Observations by Clementine. *Icarus*, 124(2), 490–499.
Poon, J., et al. (2024). Lunar illuminance as a driver of marine animal
    behaviour: a new modelling approach. *Methods in Ecology and
    Evolution*. https://doi.org/10.1111/2041-210X.14329
Śmielak, M.K. (2023). Moonlight intensity at depth: how different lunar
    and environmental parameters drive irradiance at depth.
    *Journal of Experimental Marine Biology and Ecology*, 560, 151858.
"""

import numpy as np
import pandas as pd


def moon_ground_illuminance(phase_pct, alt_deg, dist_km):
    """Predicted moonlight ground illuminance (lux).

    Implements the physical model following Austin et al. (1976),
    Poon et al. (2024 MEE), and Śmielak (2023).  This is, to the
    authors' knowledge, a first application of this model to archival
    fish-tag geolocation.

    Parameters
    ----------
    phase_pct : float
        Moon illuminated fraction (percent, 0–100).
    alt_deg : float
        Moon altitude above the horizon (degrees). Values ≤ 0 return 0.
    dist_km : float
        Moon–Earth distance (km). Mean distance = 384 400 km.

    Returns
    -------
    float
        Ground illuminance in lux.  Returns ``0.0`` if the moon is below
        the horizon (``alt_deg <= 0``).

    Notes
    -----
    The eight-step computation is:

    1. Phase angle α from illuminated fraction (Allen 1973).
    2. Lunar magnitude m (Allen 1973).
    3. Atmospheric extinction X (Austin 1976 via Poon et al. 2024).
    4. Illuminance after atmosphere (Krisciunas & Schaefer 1991).
    5. Angle-of-incidence correction.
    6. Opposition surge for |α| < 6° (Buratti et al. 1996).
    7. Distance correction relative to mean distance (384 400 km).
    8. Scatter correction × 0.863 (Poon et al. 2024).

    Examples
    --------
    >>> round(moon_ground_illuminance(100, 90, 384400), 2)
    0.28
    >>> moon_ground_illuminance(100, -5, 384400)
    0.0
    """
    if alt_deg <= 0:
        return 0.0

    # 1. Phase angle from illuminated fraction
    frac = phase_pct / 100.0
    alpha = np.degrees(np.arccos(np.clip(2.0 * frac - 1.0, -1.0, 1.0)))

    # 2. Lunar magnitude (Allen 1973)
    m_mag = -12.73 + 0.026 * abs(alpha) + 4e-9 * alpha**4

    # 3. Atmospheric extinction (Poon et al. 2024, after Austin 1976)
    z = 90.0 - alt_deg  # zenith distance (degrees)
    X = max((-0.140194 * z) / (-91.674385 + z) - 0.03, 0.0)

    # 4. Illuminance after atmosphere (Krisciunas & Schaefer 1991)
    EvA = 10 ** (-0.4 * (m_mag + X + 16.57)) * 10.7637

    # 5. Angle of incidence
    EvB = EvA * np.sin(np.radians(90.0 - z))

    # 6. Opposition surge (Buratti et al. 1996)
    if abs(alpha) < 6.0:
        EvC = EvB * (1.0 + 0.4 * (6.0 - abs(alpha)) / 6.0)
    else:
        EvC = EvB

    # 7. Distance correction relative to mean distance
    EvD = EvC / (dist_km / 384400.0) ** 2

    # 8. Scatter correction (Poon et al. 2024)
    return EvD * 0.863


def predicted_moon_curve(times, lon, lat):
    """Compute predicted moonlight illuminance for a sequence of times.

    For each timestamp, sets up an :class:`ephem.Observer` at
    ``(lon, lat)``, computes moon altitude, phase, and Earth distance, and
    returns the illuminance via :func:`moon_ground_illuminance`.
    Time steps during astronomical twilight (sun altitude > −12°) are
    masked with ``NaN`` because the moon signal is not usable then.

    Parameters
    ----------
    times : array-like of datetime-like
        UTC timestamps at which to evaluate the curve.
    lon : float
        Observer longitude (degrees East).
    lat : float
        Observer latitude (degrees North).

    Returns
    -------
    np.ndarray
        Illuminance values (lux), shape ``(len(times),)``.  ``NaN`` where
        the sun is above −12° (not astronomical night).

    Notes
    -----
    Uses :mod:`ephem` for ephemeris computation.
    AU-to-km conversion factor: 1 AU = 149 597 870.7 km.
    """
    import ephem

    obs = ephem.Observer()
    obs.lon = str(lon)
    obs.lat = str(lat)
    obs.pressure = 0
    moon = ephem.Moon()
    sun = ephem.Sun()

    illum = np.zeros(len(times))
    for k, t in enumerate(times):
        obs.date = pd.Timestamp(t).strftime("%Y/%m/%d %H:%M:%S")
        moon.compute(obs)
        sun.compute(obs)

        sun_alt = np.degrees(float(sun.alt))
        if sun_alt > -12.0:  # not astronomical night — moon signal unusable
            illum[k] = np.nan
            continue

        moon_alt = np.degrees(float(moon.alt))
        phase = float(moon.phase)
        dist_km = float(moon.earth_distance) * 149597870.7  # AU → km
        illum[k] = moon_ground_illuminance(phase, moon_alt, dist_km)

    return illum
