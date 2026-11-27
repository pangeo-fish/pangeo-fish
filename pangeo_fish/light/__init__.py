"""Light-based geolocation sub-package for pangeo-fish.

Provides solar threshold-based geolocation from archival light data
recorded by Data Storage Tags (DSTs): detects dusk/dawn threshold
crossings in the raw light record, self-calibrates the solar elevation
threshold from the first nights of deployment, evaluates event quality,
and computes the nightly timing likelihood on the cells of a HEALPix grid.

Usage
-----
::

    from pangeo_fish.light import (
        dynamic_threshold, compute_quality_flags,
        detect_twilight_events, self_calibrate_solar_threshold,
        solar_likelihood_cells, align_light_to_time, light_settings,
    )

    settings = light_settings(tag.attrs)  # from metadata.json ("tag_type")
    df["light"] = settings.transform_light(df["light"])
    df["dyn_thresh"] = dynamic_threshold(df["moon_phase"], **settings.threshold)
    pairs = detect_twilight_events(df, release_lon, release_lat)
    qdf = compute_quality_flags(df, pairs, max_depth=80, **settings.quality)
    calib = self_calibrate_solar_threshold(
        pairs, release_lon=release_lon, release_lat=release_lat
    )
    events = solar_likelihood_cells(
        pairs, qdf, emission["cell_ids"], calib.thresh_deg,
        sigma_timing=10, smooth_sigma_km=28,
    )
    emission["pdf_light"] = align_light_to_time(events, emission["time"])
"""

from pangeo_fish.light.ingest import ingest_lightloc, load_tag_csv, prepare_tag_folder
from pangeo_fish.light.quality import (
    compute_quality_flags,
    dynamic_threshold,
    twilight_quality,
)
from pangeo_fish.light.sensors import LIGHT_SETTINGS, LightSettings, light_settings
from pangeo_fish.light.solar import (
    CalibResult,
    align_light_to_time,
    compute_solar_likelihood,
    detect_twilight_events,
    light_smoothing_operator,
    pairs_from_lightloc,
    self_calibrate_solar_threshold,
    solar_likelihood_cells,
)
from pangeo_fish.light.sun import crossing_time, daily_crossings, solar_elevation

__all__ = [
    # ingest
    "load_tag_csv",
    "prepare_tag_folder",
    "ingest_lightloc",
    # quality
    "dynamic_threshold",
    "twilight_quality",
    "compute_quality_flags",
    # sensors
    "LIGHT_SETTINGS",
    "LightSettings",
    "light_settings",
    # solar
    "CalibResult",
    "detect_twilight_events",
    "pairs_from_lightloc",
    "self_calibrate_solar_threshold",
    "compute_solar_likelihood",
    "solar_likelihood_cells",
    "align_light_to_time",
    "light_smoothing_operator",
    # solar geometry
    "solar_elevation",
    "crossing_time",
    "daily_crossings",
]
