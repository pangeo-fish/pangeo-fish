"""One description of the HEALPix grid and of the CF metadata, shared by the workflow.

A dataset becomes HEALPix when it is regridded. From then on the grid is described
by the attributes of its ``cell_ids`` coordinate, following the conventions read by
xdggs and the Zarr DGGS convention:

- ``grid_name``: ``"healpix"``
- ``level``: the refinement level (int)
- ``indexing_scheme``: ``"nested"`` or ``"ring"``
- ``ellipsoid``: the reference ellipsoid, in the canonical form returned by
  :py:func:`canonical_ellipsoid`

xdggs 0.6 keeps the ellipsoid as given, xdggs 0.7 resolves a name into a dict of
parameters (or an ``Ellipsoid`` object), and some libraries only accept a name.
:py:func:`canonical_ellipsoid` maps all of these to one form: the name for a known
ellipsoid (``"WGS84"``, ``"sphere"``, ...), else a dict of numeric parameters. Every
function that needs the ellipsoid reads it through :py:func:`grid_ellipsoid`.

The variables written to Zarr carry the CF attributes of :py:data:`CF_ATTRS`
(added by :py:func:`with_cf_attrs` when missing). The stores are written by
:py:func:`pangeo_fish.storage.write_healpix_zarr` and read by
:py:func:`pangeo_fish.storage.open_healpix_zarr`.
"""

from functools import cache
from numbers import Integral

import numpy as np

CF_CONVENTIONS = "CF-1.11"

#: CF attributes of the variables produced by the workflow. Existing attributes are
#: never overwritten, so a more precise value set upstream (e.g. the units of the
#: model temperature on ``diff``) wins.
CF_ATTRS = {
    "pdf": {"long_name": "emission probability", "units": "1"},
    "pdf_tide": {"long_name": "tide emission probability", "units": "1"},
    "acoustic": {"long_name": "acoustic detection probability", "units": "1"},
    "states": {"long_name": "state probability", "units": "1"},
    "initial": {"long_name": "probability of the release position", "units": "1"},
    "final": {"long_name": "probability of the recapture position", "units": "1"},
    "mask": {"long_name": "ocean mask"},
    "ocean_mask": {"long_name": "ocean mask"},
    "diff": {"long_name": "tag temperature minus model temperature"},
    "latitude": {
        "standard_name": "latitude",
        "long_name": "latitude of the cell centre",
        "units": "degrees_north",
    },
    "longitude": {
        "standard_name": "longitude",
        "long_name": "longitude of the cell centre",
        "units": "degrees_east",
    },
    "time": {"standard_name": "time"},
    # ``cell_ids`` gets no CF attributes: xdggs reads every attribute of
    # ``cell_ids`` as a grid parameter, so only the grid attributes belong there.
}

_GRID_KEYS = ("grid_name", "level", "indexing_scheme", "ellipsoid")
_PARAMETER_ALIASES = {
    "semi_major_axis": "semimajor_axis",
    "semi_minor_axis": "semiminor_axis",
}


@cache
def _named_parameters(name):
    import healpix_geo.ellipsoid

    try:
        params = healpix_geo.ellipsoid.resolve(name)
    except Exception:
        return None
    return {k: float(v) for k, v in dict(params).items() if k != "name"}


def _same_parameters(a, b):
    keys = set(a) & set(b)
    return bool(keys) and all(np.isclose(a[k], b[k], rtol=1e-12) for k in keys)


def canonical_ellipsoid(ellipsoid):
    """Return the canonical form of a HEALPix ``ellipsoid``.

    Parameters
    ----------
    ellipsoid : None, str, dict or xdggs ellipsoid object
        As found in the ``cell_ids`` attributes or in ``ds.dggs.grid_info``.
        ``None`` stands for the sphere used by default.

    Returns
    -------
    str or dict
        The name (e.g. ``"WGS84"``, ``"sphere"``) when the ellipsoid is a named one
        with its standard parameters, else a dict of its numeric parameters
        (``semimajor_axis`` and ``inverse_flattening``, or ``radius``).
    """
    if ellipsoid is None:
        return "sphere"
    if isinstance(ellipsoid, str):
        if _named_parameters(ellipsoid) is None:
            raise ValueError(f"unknown ellipsoid name {ellipsoid!r}")
        return ellipsoid
    if hasattr(ellipsoid, "to_dict"):
        ellipsoid = ellipsoid.to_dict()
    if not hasattr(ellipsoid, "items"):
        raise TypeError(f"cannot interpret {ellipsoid!r} as an ellipsoid")

    params = {
        _PARAMETER_ALIASES.get(key, key): value for key, value in ellipsoid.items()
    }
    name = params.pop("name", None)
    params = {k: float(v) for k, v in params.items() if v is not None}
    if name is not None:
        named = _named_parameters(name)
        if named is not None and (not params or _same_parameters(params, named)):
            return name
    if not params:
        raise ValueError(f"cannot interpret {ellipsoid!r} as an ellipsoid")
    return params


def grid_ellipsoid(obj, default="sphere"):
    """Return the canonical ellipsoid of a HEALPix Dataset, DataArray or ``cell_ids``.

    The xdggs index wins when the object is decoded; else the ``ellipsoid``
    attribute of ``cell_ids`` is used, else ``default``.
    """
    ellipsoid = None
    try:
        ellipsoid = obj.dggs.grid_info.ellipsoid
    except Exception:
        pass
    if ellipsoid is None:
        try:
            attrs = obj["cell_ids"].attrs
        except Exception:
            attrs = getattr(obj, "attrs", {})
        ellipsoid = attrs.get("ellipsoid", default)
    return canonical_ellipsoid(ellipsoid)


def grid_attrs(ds):
    """Return the canonical grid attributes of the HEALPix dataset ``ds``.

    Returns ``None`` when ``ds`` has no ``cell_ids`` or no known level.
    """
    if "cell_ids" not in ds.variables:
        return None
    attrs = dict(ds["cell_ids"].attrs)
    try:
        info = ds.dggs.grid_info
    except Exception:
        info = None
    if info is not None:
        attrs["level"] = info.level
        attrs["indexing_scheme"] = getattr(info, "indexing_scheme", None) or attrs.get(
            "indexing_scheme"
        )
        if getattr(info, "ellipsoid", None) is not None:
            attrs["ellipsoid"] = info.ellipsoid
    if attrs.get("level") is None:
        return None
    level = attrs["level"]
    if not isinstance(level, Integral):
        level = int(np.asarray(level).item())
    return {
        "grid_name": "healpix",
        "level": int(level),
        "indexing_scheme": attrs.get("indexing_scheme") or "nested",
        "ellipsoid": canonical_ellipsoid(attrs.get("ellipsoid")),
    }


def with_grid_attrs(ds):
    """Return ``ds`` with the canonical grid attributes on ``cell_ids``.

    Other attributes of ``cell_ids`` are kept. ``ds`` is returned unchanged when it
    has no HEALPix grid.
    """
    attrs = grid_attrs(ds)
    if attrs is None:
        return ds
    ds = ds.copy()
    ds["cell_ids"].attrs.update(attrs)
    return ds


def with_cf_attrs(ds):
    """Return ``ds`` with the CF attributes of :py:data:`CF_ATTRS` where missing.

    Also sets the global ``Conventions`` attribute. Existing attributes win.
    """
    ds = ds.copy()
    for name, attrs in CF_ATTRS.items():
        if name in ds.variables:
            current = ds[name].attrs
            ds[name].attrs = {
                k: v for k, v in attrs.items() if k not in current
            } | dict(current)
    if "Conventions" not in ds.attrs:
        ds.attrs["Conventions"] = CF_CONVENTIONS
    return ds
