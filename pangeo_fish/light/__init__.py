"""Light-based geolocation sub-package for pangeo-fish.

Provides solar threshold-based geolocation from archival light data
recorded by Data Storage Tags (DSTs): detects dusk/dawn threshold
crossings in the raw light record, self-calibrates the solar elevation
threshold from the first nights of deployment, evaluates event quality,
and computes per-night Gaussian likelihood maps on a regular lat/lon grid.

Usage
-----
::

    from pangeo_fish.light import (
        dynamic_threshold, compute_quality_flags,
        detect_twilight_events, self_calibrate_solar_threshold,
        compute_solar_likelihood,
    )
"""

from pangeo_fish.light.ingest import ingest_lightloc, load_tag_csv, prepare_tag_folder
from pangeo_fish.light.quality import (
    compute_quality_flags,
    dynamic_threshold,
    twilight_quality,
)
from pangeo_fish.light.solar import (
    CalibResult,
    compute_solar_likelihood,
    detect_twilight_events,
    pairs_from_lightloc,
    self_calibrate_solar_threshold,
)

__all__ = [
    # ingest
    "load_tag_csv",
    "prepare_tag_folder",
    "ingest_lightloc",
    # quality
    "dynamic_threshold",
    "twilight_quality",
    "compute_quality_flags",
    # solar
    "CalibResult",
    "detect_twilight_events",
    "pairs_from_lightloc",
    "self_calibrate_solar_threshold",
    "compute_solar_likelihood",
]
