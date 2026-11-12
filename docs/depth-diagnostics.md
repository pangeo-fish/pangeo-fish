# Tag depth versus model bathymetry

The temperature emission (`compute_diff`) compares the tag temperature with
the model temperature at the tag depth. When the tag goes deeper than the model
seabed, two things can happen:

- with `relative_depth_threshold = 0` (default in the demos), the comparison
  uses the deepest model level, and the contradiction is ignored;
- with `relative_depth_threshold = r > 0`, a cell is removed at a time step
  when its bathymetry (`H0 + XE`) is lower than `r` times the maximum tag depth
  of that time step.

The second option is useful when the model bathymetry is accurate, because it
removes cells where the fish cannot be. It is harmful when the model grid is too
coarse for the area. In estuaries and narrow channels, a 2–3 km model cell
holds both a deep channel and shallow flats, so the model depth is much
lower than the depth the fish actually reaches.

## Diagnostic

`diagnose_depth_vs_bathymetry` computes, for each model time step, the maximum
tag depth and the share of valid cells that are too shallow. It can do the same
for an independent bathymetry such as GEBCO.
`pangeo_fish.bathy.sample_max_depth` samples GEBCO at the model cells and keeps the
deepest value within a box of the model resolution:

```python
import xarray as xr

from pangeo_fish.bathy import sample_max_depth
from pangeo_fish.helpers import diagnose_depth_vs_bathymetry

url = "https://data-taos.ifremer.fr/demo/database/gebco_2024_new.zarr/"
gebco = xr.open_zarr(url)["elevation"]
gebco_depth = sample_max_depth(
    gebco, diff["latitude"], diff["longitude"], window=1 / 36
)

diagnostic = diagnose_depth_vs_bathymetry(
    tag_log=tag_log,
    bathymetry=diff["H0"],
    time=diff["time"],
    reference_bathymetry=gebco_depth.where(diff["ocean_mask"]),
    relative_depth_threshold=1.0,
    # optional: restrict to the area of interest, e.g. an estuary
    # mask=area_mask,
)
diagnostic[["model_too_shallow", "reference_too_shallow"]].to_dataframe()
```

Variables (along `time`):

- `tag_max_depth`: deepest tag record of the time step;
- `model_too_shallow` / `reference_too_shallow`: share of valid cells whose
  depth is lower than `relative_depth_threshold * tag_max_depth`;
- `model_only_too_shallow`: share of valid cells that are too shallow in the
  model but deep enough in the reference, i.e. the cells a threshold would
  remove because of the model bathymetry only;
- `model_max_depth` / `reference_max_depth`: deepest valid cell.

A `UserWarning` is emitted when, at some time steps, every valid cell is too
shallow in the model.

## When to use what

- `model_too_shallow` low and close to `reference_too_shallow`: the model
  bathymetry is consistent with GEBCO, and `relative_depth_threshold` (e.g.
  0.8) is a cheap way to use the depth information.
- `model_too_shallow` close to 1 in the area where the fish is expected, while
  `reference_too_shallow` is clearly lower (large `model_only_too_shallow`):
  the model is too coarse there. Keep `relative_depth_threshold = 0` and add
  the bathymetry evidence of `pangeo_fish.bathy`
  (`compute_healpix_histogram_region_bin_size` and `batch_compute_pdf_bathy`),
  which uses the GEBCO depth distribution inside each HEALPix cell.
- Both close to 1: check the tag depths first, e.g. the pressure unit and offset
  (see #67).

Example: twaite shad A19845 (`shad_scheldt_dst`, IBI, HEALPix level 11) in the
Western Scheldt (98 cells). The tag reaches 13–40 m per day. The IBI depth is
at most 17.3 m, so all cells are too shallow at 23 of 36 days. With GEBCO (deepest
value per cell, up to 49 m) no day has all cells too shallow.
