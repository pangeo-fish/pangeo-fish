# Dry cells in hourly shelf models

In tidal shelf models, the water column of a sea cell is `H0 + XE`: the static
depth plus the sea surface height. pangeo-fish calls a sea cell **dry** at time
`t` when `H0 + XE <= min_water_depth` (an argument of `compute_diff`; `0.0` means
strictly dry).

## Where it happens

Only the Copernicus Marine NWSHELF hourly product (`cmems_mod_nws_phy-tem_anfc_1.5km-3D_PT1H-i`)
shows dry cells in practice. The NWS model has no wetting and drying: it runs with a
10 m minimum depth and the fields are interpolated back to the real depths
(NWS Product User Manual 3.0, §2b). The distributed static depth `H0` is the real
bathymetry, so on tidal flats `H0 + XE` falls below zero at low tide while the
temperature stays finite. Dry cells can therefore not be detected from `TEMP`.

| Area (NWS hourly) | Day | Dry (cell, time) pairs | Max dry share of sea cells |
| --- | --- | --- | --- |
| Mont-Saint-Michel bay | 2025-03-31 (spring tide) | 179 | 24 / 356 |
| Mont-Saint-Michel bay | 2025-04-06 (neap tide) | 4 | 1 / 356 |
| Western Scheldt | 2025-03-31 | 351 | 38 / 163 |
| Western Scheldt | 2025-04-06 | 427 | 37 / 163 |

IBI (hourly and daily) and GLORYS have a minimum static depth of 5 m and 7 m and
show no dry cells.

## What `compute_diff` does

- `min_water_depth=None` (default): the values are unchanged. If there are dry
  cells, a `DryCellWarning` (a `UserWarning`) reports how many.
- `min_water_depth=0.0`: the `diff` of the dry (cell, time) pairs becomes NaN, which
  gives them a probability of 0 later on. A positive value also excludes nearly
  dry cells. `diff.attrs["min_water_depth"]` records the value. Older pangeo-fish
  versions silently ignore the argument, so check this attribute.
- A time step where no wet cell with a valid `diff` would remain is never masked
  (the warning lists it).
- `H0`, `XE` and `ocean_mask` are not modified.

```python
diff, _ = compute_diff(
    reference_model=model,
    tag_log=tag["dst"].ds,
    relative_depth_threshold=0,
    min_water_depth=0.0,
)
```

With daily products this is a no-op (no dry cells), apart from the attribute.

## Things to know

- The check runs on the native model grid, before regridding. Regridding spreads
  NaN by about one ring, so wet neighbours of dry cells are also lost (in the
  Western Scheldt at low spring tide, the wet HEALPix cells drop by about 20 % at
  level 12 and more at levels 11 and 13). Level 12 (about 1.5 km) is closest to
  the native NWS grid.
- `count_dry_cells` on a regridded dataset does not detect dry cells, because the
  regridding clips negative values.
- If another likelihood (for example an acoustic detection buffer) only covers dry
  cells at some time step, the product of the probabilities is zero there. Check
  `combined["pdf"].sum(spatial_dims)` before running the HMM.
- Wet cells shallower than 10 m are also interpolated from the model's 10 m water
  column, and the NWS manual advises against using them directly where they are
  widespread. Use a positive `min_water_depth` to be stricter.
- Levels above the instantaneous surface (`depth + XE < 0`) are kept as before.

`pangeo_fish.drycells.count_dry_cells(model)` gives the number of dry cells per
time step for a loaded model.
