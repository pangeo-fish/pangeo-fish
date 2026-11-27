"""Light settings per type of tag, chosen from the tag metadata.

The thresholds of the twilight detection and of the quality filter depend on
how the tag records light: raw counts on a Wildlife Computers archive record,
``log1p`` of the counts on a LoggLaw C6, and so on. Instead of setting them
by hand in every notebook, the tag says what it is in its ``metadata.json``
(the attributes of the root node of :func:`pangeo_fish.io.open_tag`) and
:func:`light_settings` returns the matching values:

.. code-block:: json

    {"tag_type": "wc_archival"}

``tag_type`` uses the names of :func:`pangeo_fish.light.ingest.load_tag_csv`
(``prepare_tag_folder`` already writes it). A ``"light"`` entry in the
metadata overrides single values, for example
``{"tag_type": "wc_archival", "light": {"min_amplitude": 30}}``.

Only the types whose values have been used and checked on real records are
listed in :data:`LIGHT_SETTINGS`. For any other type :func:`light_settings`
raises, so that the values are chosen on purpose and added here.

The largest depth allowed during a twilight (``max_depth`` of
:func:`~pangeo_fish.light.quality.compute_quality_flags`) is a property of
the animal or of the deployment, not of the sensor, and is not part of these
settings.
"""

from dataclasses import dataclass, field, fields, replace

import numpy as np

_QUALITY_KEYS = (
    "window_h",
    "slope_threshold",
    "marge_threshold",
    "min_amplitude",
    "min_samples",
)


@dataclass(frozen=True)
class LightSettings:
    """Light-processing values for one type of tag.

    Attributes
    ----------
    tag_type : str
        Name of the type of tag, as in ``metadata.json``.
    transform : {None, "log1p"}
        Transform applied to the recorded light before thresholding
        (:meth:`transform_light`).
    threshold_base, threshold_amp : float
        ``base`` and ``amp`` of
        :func:`~pangeo_fish.light.quality.dynamic_threshold`, on the
        transformed scale.
    window_h, slope_threshold, marge_threshold, min_amplitude, min_samples
        Keyword arguments of
        :func:`~pangeo_fish.light.quality.compute_quality_flags`, on the
        transformed scale. ``slope_threshold=None`` turns the direction
        filter off.
    description : str
        What the values were taken from.
    """

    tag_type: str
    transform: str | None
    threshold_base: float
    threshold_amp: float
    window_h: float
    slope_threshold: float | None
    marge_threshold: float
    min_amplitude: float
    min_samples: int
    description: str = field(default="", compare=False)

    @property
    def quality(self):
        """Keyword arguments for :func:`~pangeo_fish.light.quality.compute_quality_flags`."""
        return {key: getattr(self, key) for key in _QUALITY_KEYS}

    @property
    def threshold(self):
        """Keyword arguments for :func:`~pangeo_fish.light.quality.dynamic_threshold`."""
        return {"base": self.threshold_base, "amp": self.threshold_amp}

    def transform_light(self, light):
        """Apply :attr:`transform` to the recorded light."""
        if self.transform is None:
            return light
        if self.transform == "log1p":
            return np.log1p(light)
        raise ValueError(f"unknown light transform {self.transform!r}")


LIGHT_SETTINGS = {
    "wc_archival": LightSettings(
        tag_type="wc_archival",
        transform=None,
        threshold_base=60.0,
        threshold_amp=20.0,
        window_h=1.0,
        slope_threshold=25.0,
        marge_threshold=8.0,
        min_amplitude=40.0,
        min_samples=10,
        description=(
            "Wildlife Computers MiniPAT, recovered archive record (Light Level, "
            "raw counts); values of the bluefin tuna 18P0430 and 20P0204"
        ),
    ),
    "logglaw_c6": LightSettings(
        tag_type="logglaw_c6",
        transform="log1p",
        threshold_base=3.0,
        threshold_amp=0.3,
        window_h=1.5,
        slope_threshold=None,
        marge_threshold=0.3,
        min_amplitude=0.5,
        min_samples=10,
        description=(
            "Biologging Solutions LoggLaw C6, log1p of the raw counts; values of "
            "the six-logger null test at Maizuru"
        ),
    ),
}


def light_settings(metadata, **overrides):
    """Light settings of a tag, from its metadata.

    Parameters
    ----------
    metadata : mapping or str
        The tag metadata (``tag.attrs`` of :func:`pangeo_fish.io.open_tag`,
        or the content of ``metadata.json``) with a ``tag_type`` entry, or
        the ``tag_type`` itself. An optional ``"light"`` mapping in the
        metadata overrides single values.
    **overrides
        Values that override both the defaults of the type and the metadata.

    Returns
    -------
    LightSettings

    Raises
    ------
    KeyError
        If the metadata has no ``tag_type``, or no settings are known for it.
    """
    if isinstance(metadata, str):
        metadata = {"tag_type": metadata}
    tag_type = metadata.get("tag_type")
    if tag_type is None:
        raise KeyError(
            "the tag metadata has no 'tag_type'; add it to metadata.json "
            f"(one of {sorted(LIGHT_SETTINGS)}) or pass the light values by hand"
        )
    key = str(tag_type).lower()
    if key not in LIGHT_SETTINGS:
        raise KeyError(
            f"no light settings for tag_type {tag_type!r}; known types: "
            f"{sorted(LIGHT_SETTINGS)}. Choose the values for this tag and add "
            "them to pangeo_fish.light.sensors.LIGHT_SETTINGS"
        )
    custom = dict(metadata.get("light") or {})
    custom.update(overrides)
    known = {f.name for f in fields(LightSettings)} - {"tag_type"}
    unknown = set(custom) - known
    if unknown:
        raise TypeError(f"unknown light settings: {sorted(unknown)}")
    return replace(LIGHT_SETTINGS[key], **custom)
