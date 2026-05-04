"""Light-based geolocation sub-package for pangeo-fish.

Provides two complementary methods for inferring fish position from
archival light data recorded by Data Storage Tags (DSTs):

**Part A — Solar threshold (``pangeo_fish.light.solar``)**
    Detects dusk/dawn threshold crossings in the raw light record,
    self-calibrates the solar elevation threshold from the first nights of
    deployment, evaluates event quality, and computes per-night Gaussian
    likelihood maps on a regular lat/lon grid.

**Part B — Lunar template fit (``pangeo_fish.light.lunar``)**
    Identifies nights during which the fish was shallow and the moon was
    sufficiently bright, computes the Spearman correlation between the
    observed sub-surface light curve and the physically predicted moonlight
    illuminance at every grid pixel, and converts correlations to proper
    likelihood maps.

The physical moonlight model follows Austin et al. (1976), Poon et al.
(2024 MEE), and Śmielak (2023) and is implemented in
``pangeo_fish.light.physics``.  To the authors' knowledge, this is the
first application of this physical model to archival fish-tag geolocation.

Usage
-----
::

    from pangeo_fish.light import (
        dynamic_threshold, compute_quality_flags,
        detect_twilight_events, self_calibrate_solar_threshold,
        compute_solar_likelihood,
        find_usable_moon_nights, compute_lunar_correlation_maps,
        correlation_to_likelihood, lunar_likelihood_to_dataarray,
        moon_ground_illuminance, predicted_moon_curve,
    )

References
----------
Poon, J., et al. (2024). Lunar illuminance as a driver of marine animal
    behaviour: a new modelling approach. *Methods in Ecology and
    Evolution*. https://doi.org/10.1111/2041-210X.14329
Śmielak, M.K. (2023). Moonlight intensity at depth: how different lunar
    and environmental parameters drive irradiance at depth.
    *Journal of Experimental Marine Biology and Ecology*, 560, 151858.
"""

from pangeo_fish.light.ingest import load_tag_csv, prepare_tag_folder
from pangeo_fish.light.lunar import (
    apply_cloud_weighting,
    compute_lunar_correlation_maps,
    correlation_to_likelihood,
    find_usable_moon_nights,
    lunar_likelihood_to_dataarray,
)
from pangeo_fish.light.physics import moon_ground_illuminance, predicted_moon_curve
from pangeo_fish.light.quality import (
    compute_quality_flags,
    dynamic_threshold,
    twilight_quality,
)
from pangeo_fish.light.solar import (
    CalibResult,
    compute_solar_likelihood,
    detect_twilight_events,
    self_calibrate_solar_threshold,
)

__all__ = [
    # ingest
    "load_tag_csv",
    "prepare_tag_folder",
    # quality
    "dynamic_threshold",
    "twilight_quality",
    "compute_quality_flags",
    # physics
    "moon_ground_illuminance",
    "predicted_moon_curve",
    # solar
    "CalibResult",
    "detect_twilight_events",
    "self_calibrate_solar_threshold",
    "compute_solar_likelihood",
    # lunar
    "find_usable_moon_nights",
    "compute_lunar_correlation_maps",
    "correlation_to_likelihood",
    "lunar_likelihood_to_dataarray",
    "apply_cloud_weighting",
]
