# How the likelihoods are built

This page describes what the code does, step by step, for each term that enters the HMM.
It is written from the functions in `pangeo_fish`, not from the papers.
For the tide terms, see {doc}`tide`.
For the movement model, see {doc}`choosing-conv-method`.
For the input files, see {doc}`format`.

All grids are HEALPix grids on the WGS84 ellipsoid (the default `ellipsoid="WGS84"`).

## 1. Temperature (emission) likelihood

The temperature term is built in three steps: `compute_diff`, then `regrid_dataset_hpresample`, then `compute_emission_pdf`.

### `compute_diff`

1. The tag log is cut into the time bins of the reference model.
   The bins come from the bounds of the model `time` coordinate (`reshape_by_bins`).
   Every tag record that falls in a bin belongs to that bin.
2. In each bin and each model cell, levels with a missing temperature or depth are dropped.
3. The depth of a level is `dynamic_depth`, the depth below the instantaneous sea surface.
   A level above the surface (negative `dynamic_depth`, for example at low tide) is moved to depth 0.
4. For each tag record with a valid `pressure`, the model level with the closest depth is chosen (`diff_z`).
   There is no vertical interpolation.
   A record with a missing temperature gives no residual.
5. The residual of a record is `tag temperature - model temperature` at that level.
6. `diff` is the mean of the residuals of the bin.
   It is signed. A bin without any valid record gives `NaN`.
7. Dry cells are masked as described in {doc}`dry-cells` (`min_water_depth`).

The tag `pressure` column is used as a depth in metres, as it is.
There is no dbar to metre conversion.
Convert the column yourself if your sensor does not record metres of water.

`relative_depth_threshold` is no longer used.
`compute_diff` stores it in the attributes and raises a `FutureWarning` if it is not 0.

### Regridding

`regrid_dataset_hpresample` moves the signed mean residual `diff` to the HEALPix cells.
The likelihood is computed afterwards, so what is regridded is the residual, not a probability.
The values are not clipped, so `diff` can stay negative.
Sea cells are chosen with `mask_threshold` (default `0.5`), and land cells get `NaN`.

### `compute_emission_pdf`

- `pdf` is the normal density of `diff` with mean 0 and standard deviation `differences_std` (`scipy.stats.norm.pdf`).
  `differences_std` has the unit of `diff`, the model temperature unit.
- `initial` is a Gaussian around the `release` position.
  Its width is `initial_std`, in radians.
  It is a function of the angular distance between each cell centre and the centre of the cell that holds the position.
- `final` is the same for the `fish_death` position, with width `recapture_std`.
  It is left out when the latitude and longitude of `fish_death` are both `NA`.
- `mask` is the ocean mask.
- With `check_positions=True`, a warning is raised when `initial` or `final` has almost no mass on ocean cells.

`pdf` itself is not normalised.
`initial` is the starting distribution of the HMM.
`final` is multiplied into the last time step when the terms are combined (see section 4).

## 2. Acoustic likelihood

`compute_acoustic_pdf` calls `emission_probability` with `nondetections="mask"`.
If the tag has no `acoustic` or no `stations`, `emission_probability` returns an empty dataset.

1. Detections are counted per receiver (`deployment_id`) in each time bin of the emission grid (`count_detections`).
   A `deployment_id` that is missing from `stations` raises a `ValueError`.
   The detection times and the bins are first converted to `datetime64[ns]` (pandas 3 reads the CSV times as `datetime64[us]`).
2. The centre of a receiver is `deploy_longitude`, `deploy_latitude`.
   The recover position is not used.
3. The buffer of a receiver is the set of cells covered by a disc of radius `receiver_buffer` around that centre.
   It is a 0/1 mask per receiver.
   The radius in metres is turned into an angle with a fixed radius of 6371 km (`buffer_points_cells`).
4. In a time bin with detections, the weight of a cell is the sum over receivers of `mask * number of detections`.
   Cells in the buffer of several detecting receivers add up.
   The buffers are not intersected.
5. The result is normalised over the cells of each time step, and land cells are set to `NaN`.

In a time bin without any detection, `nondetections` decides:

| Value      | Weight in a time bin without detection                                                                                                                                                                                     |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"mask"`   | Cells inside the buffer of a receiver that was active during the whole bin get `NaN`, so they are removed when the terms are combined (section 4). Other cells share the weight equally. `compute_acoustic_pdf` uses this. |
| `"ignore"` | All cells have the same weight.                                                                                                                                                                                            |

A receiver is active in a bin when the whole bin lies between its `deploy_time` and `recover_time`.
A bin where the receiver was deployed or recovered only part of the time does not count, so its buffer is not removed there.
A receiver that never covers a whole bin is left out of the mask (`create_masked_fill_map`), so its buffer is never removed.
This includes a receiver whose `recover_time` is missing (`NaT`), for example one still in the water at the end of the study.

## 3. Bathymetry likelihood

The term has two parts.
The first, `compute_healpix_histogram_region_bin_size`, runs once per grid.
The second, `batch_compute_pdf_bathy`, runs per tag.

### Per cell: exceedance probability

For each HEALPix cell, the function collects the GEBCO points with `elevation < 0`.
Land points and `NaN` are left out.
Each point gives a depth `-elevation - depth_offset`, spread as a normal distribution with its `stdev`.
The depths are added into bins.
The result `bathy_pixel_hist` is the probability that the seabed is at least `d` metres deep, for every bin edge `d`.
It is 1 at the shallowest edge and decreases with depth.
The edges start 10 bins above the surface.
With `max_depth_m`, the last edge is `max_depth_m` and holds the share of all deeper seabed.

| Parameter         | Default   | Meaning                                                                                |
| ----------------- | --------- | -------------------------------------------------------------------------------------- |
| `depth_bin_size`  | `1` m     | Bin width. The likelihood is linear inside a bin.                                      |
| `max_depth_m`     | `None`    | Depth of the last edge. Give this or `nb_depth_bins`.                                  |
| `depth_offset`    | `0` m     | Clearance of the fish above the seabed. A negative value lets the fish go below GEBCO. |
| `default_stdev_m` | `1` m     | Uncertainty used where `stdev` is missing or not finite. The minimum used is 0.001 m.  |
| `ellipsoid`       | `"WGS84"` | Ellipsoid of the HEALPix grid. It must match the one of `diff-regridded.zarr`.         |

### Per time step

`batch_compute_pdf_bathy` reads `XE` from `diff-regridded.zarr`.
For each time bin and each cell:

1. The deepest `pressure` of the bin is taken, as dbar read as metres.
2. The depth below mean sea level is `deepest - XE`.
3. The likelihood is `bathy_pixel_hist` at that depth, with linear interpolation between edges.
   A depth above the first edge gives the first value.
   A depth below the last edge gives the last value.
4. The value is never lower than `1e-14`.

Fallbacks and guards:

- A bin without any valid pressure gives 1 in every sea cell.
- Land cells (`XE` is `NaN`) give `NaN`.
- A sea cell without a GEBCO water point uses the histogram of the nearest histogram cell, and a `UserWarning` reports how many cells and how far.
- A pressure above 12000 raises a `ValueError` (likely Pa instead of dbar).
- A histogram from another grid, or from an older version, raises a `ValueError`.
- Every tag time step must exist in `diff-regridded.zarr`.

## 4. Combining the terms

`normalize_pdf` calls `combine_emission_pdf`.
The demos use it.

1. Every data variable except `initial`, `final` and `mask` (and except those in `excluded_pdf`) is a term.
   Typical terms are `pdf`, `acoustic`, `pdf_bathy`, `pdf_tide` and `pdf_light`.
2. The terms are multiplied with `skipna=False`.
   A `NaN` in one term makes the product `NaN` in that cell and time step.
3. `final` is multiplied into the last time step.
4. The product is divided by its sum over the cells, for each time step.
   `NaN` cells are skipped in that sum.
5. The result is stored as `pdf`.

`normalize_pdf` warns when the input `pdf` (the temperature term) has no value at some time steps.
This check runs before the product, so an empty product caused by another term is not reported there.
It does not stop there.
`optimize_pdf` calls an emission check first.
That check raises a `ValueError` when a time step has no positive mass (`NaN`, 0 or negative everywhere), inside `mask` if the dataset has one.

`combine_pdfs` is an older helper.
It combines the terms with `combine_emission_pdf`.
When, in every cell of a time step, the acoustic value is 0 or the temperature `pdf` is `NaN`, it replaces `pdf` by the acoustic values and warns.

### How the terms are aligned

- The acoustic term is built on the cells of the temperature dataset.
  The demo notebooks drop the `cell_ids` index and merge, so cells are matched by position, and time by label.
  Build all terms on the same grid.
- The bathymetry term is selected with `sel(cell_ids=..., time=...)` before the merge, so it is matched by `cell_ids` and by time.
- The tide term is matched by `cell_ids` and by the day label in `merge_tide_with_total`.
  The grid attributes (`grid_name`, `level`, `indexing_scheme`, `ellipsoid`) must agree, or a `ValueError` is raised.
  Cells or days that the tide store does not cover get a constant factor and a `UserWarning`.
  In that case the tide term has no effect there.
  The complete demo then clips `pdf_tide` at 0 (`clip(min=0)`) before combining.
  Interpolation can give small negative values, and a time step where every value is 0 or negative counts as having no mass.

## 5. Options and defaults

### `optimize_pdf`

| Option         | Default     | Effect                                                                                                                                                               |
| -------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `single_behav` | `True`      | Multi-sigma only, not multivariate. `True`: one sigma per predictor value. `False`: one sigma per continuous segment.                                                |
| `multivariate` | `False`     | Multi-sigma only. `True`: all sigmas are optimised together on the whole sequence, and `single_behav` is ignored.                                                    |
| `final`        | `False`     | `True`: sigma is fitted on the final position instead of the forward likelihood. The upper bound is then 0.01. Not allowed with multi-sigma (`NotImplementedError`). |
| `conv_method`  | `"FFTConv"` | See {doc}`choosing-conv-method`.                                                                                                                                     |

Multi-sigma mode starts when the dataset has a `predictor_index` variable, one integer per time step.
`stamp_parameter_indices` builds it from a list of times (nearest time step).
`stamp_parameter_indices_from_mask` builds it from a 0/1 mask, matched by position.
In multi-sigma mode, `ds.attrs["sigma_bounds"]` can give one speed range per behaviour, as a dictionary `{predictor value: (minimum speed, maximum speed)}`.
`stamp_parameter_indices_from_mask(..., sigma_bounds_dict=...)` sets it.
Every predictor value present in the data needs an entry, or `optimize_pdf` raises a `ValueError`; entries for values absent from the data are ignored.
Only the maximum speed is used.
Without `sigma_bounds`, every behaviour uses the upper bound derived from `maximum_speed`.
Sigma is searched in radians, from `1e-4` to `maximum_speed * time step * adjustment_factor / earth_radius`.
By default `earth_radius` is the semi-major axis of the grid ellipsoid (6378.137 km for WGS84); it can be set with `earth_radius=`.
In multi-sigma mode with `sigma_bounds`, each behaviour has its own upper bound, and the lower bound stays `1e-4`.

### Tide likelihood

| Name               | Default                                   | Where                                                                    |
| ------------------ | ----------------------------------------- | ------------------------------------------------------------------------ |
| `tide_sigma_m`     | `2.0` (demo)                              | Model error of `still_tag_tide_likelihood` (no default in the function). |
| `tide_min_samples` | `6` (demo), `12` (function `min_samples`) | Minimum number of valid hourly means for a day to count.                 |
| `tideFL`           | `25` h                                    | `detect_tide`: window of the tide detection (at least 12.42 h).          |
| `tideLV`           | `[0.42, 0.85, 0.6]`                       | `detect_tide`: maximum RMSE (m), minimum R², minimum amplitude (m).      |
| `behavFL`          | `25` h                                    | `detect_tide`: window of the relaxed fit behind `behav_found`.           |
| `behavLV`          | `[0.42, 0.70, 0.2]`                       | `detect_tide`: thresholds of the relaxed fit.                            |

`detect_tide` resamples the pressure to 10 minutes; a coarser time step raises a `NotImplementedError`.
`tide_sigma_m` is read only by `tide_method = "healpix"`.
See {doc}`tide` for what the two tide methods assume.

## 6. Light likelihood

The light term uses the times at which the light record crosses a light threshold at dawn and dusk.
The functions are in `pangeo_fish.light`.
The solar geometry is in `pangeo_fish.light.sun`.
It is computed for many positions at once, without a loop over cells.
It is the low-precision solar position of the NOAA calculator: elevation of the centre of the Sun, without refraction.
Over 1995--2026 it agrees with `ephem` to 0.02 degree of elevation and 0.4 minute of crossing time.

### Detection: `detect_twilight_events`

1. A rising is a record where the light goes from at most the threshold to above it.
   A setting is the reverse.
   The threshold is `dyn_thresh` per row, or the moon-phase threshold of `dynamic_threshold`, or `thresh_light`.
   With `use_fixed_threshold=True` it is the largest value of the per-row threshold.
2. For each local solar day, the rising and the setting at the release position are predicted for the elevation `detection_elevation` (default -10 degrees).
3. The threshold crossing nearest to each prediction is kept if it is less than `window_h` (default 2.5 h) away.
4. A day gives a pair `(t_rise, t_set)` only if both crossings are kept.

The day is the local solar day, centred on the solar noon at the release longitude.
East of about 90 degrees East the rising is on the previous UTC date (about 21:00 UTC for a sunrise in Japan).

### Quality: `compute_quality_flags`

Each event is checked in a window of `window_h` hours (default 1 h) around it:

| Filter    | Condition                                                                                                       | Parameter                                                  |
| --------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Direction | mean of the last 10 records minus mean of the first 10 is above the value for a rising, below minus it for a setting | `slope_threshold` (default 25, `None` turns it off) |
| Margin    | peak light minus the mean threshold is above the value                                                          | `marge_threshold` (default 8)                              |
| Depth     | largest pressure is below the value, in dbar                                                                    | `max_depth` (default 80)                                   |
| Amplitude | light range (max - min) is above the value                                                                      | `min_amplitude` (default 40)                               |
| Samples   | number of records in the window is at least the value                                                           | `min_samples` (default 10)                                 |

A night is `good` when both events pass, `partial` when one passes and `bad` when none passes.
The direction filter is told which event is a rising.
`twilight_quality` called alone without `direction` still guesses it from the UTC hour (before 12:00 is a rising), which is wrong far from the Greenwich meridian.
The defaults are for raw light counts; a log-transformed sensor needs other values.

### Self-calibration: `self_calibrate_solar_threshold`

The solar elevation is computed at each crossing of the first `n_nights` pairs, at the release position.
The median is `thresh_deg`, the elevation at which the tag sees its threshold.
With a recapture position, the same median over the last `n_nights_end` pairs gives `thresh_deg_end`, as a check.

### Likelihood: `solar_likelihood_cells`

For each night and each HEALPix cell:

1. The crossings of `thresh_deg` are predicted at the cell centre, on the ellipsoid of `cell_ids`.
   A prediction more than `window_h` (default 6 h) from the observed crossing counts as missing, and so does one that does not exist (polar day or night).
2. Each valid event of the night gives `exp(-dt**2 / (2 * sigma_timing**2))`, with `dt` in minutes.
   The likelihood is the product.
3. A `bad` night gives 1 in every cell.
4. A cell without any valid prediction gets `lh_floor` (default `1e-30`).
   On a `good` night both predictions are needed: a cell with only one gets `lh_floor`.
5. Each night is divided by its maximum.

The time of a night is its midpoint (local noon, UTC).
There is no spatial smoothing: the likelihood is evaluated at the cell centres, not on an intermediate latitude/longitude grid.

`compute_solar_likelihood` gives the same likelihood on a regular latitude/longitude grid, smoothed with a Gaussian filter of `smooth_sigma` grid steps.
It is the output of the notebooks of pull request 267, which then regridded it to HEALPix.

### Time steps: `align_light_to_time`

`align_light_to_time(nightly, emission["time"])` gives `pdf_light` on the time steps of the emission.

| `how`                | A night goes to                                                         |
| -------------------- | ----------------------------------------------------------------------- |
| `"date"` (default)   | the step labelled 00:00 of the UTC date of its midpoint (the local day) |
| `"rise_date"`        | the step labelled 00:00 of the UTC date of its rising, as in pull request 267 |
| `"bins"`             | the step whose bin, built as in `compute_diff`, contains its midpoint   |

Use `"date"` for daily steps labelled at 00:00 and `"bins"` for steps shorter than a day.
With daily steps, `"bins"` puts the bin edges at 12:00 UTC, near local noon in Europe and Africa, so two nights can share a step and leave the next one empty.
A step without any night gets `fill_value` (default 1), which is neutral.
Nights in the same step are multiplied and divided by the maximum again.
`n_nights` counts the nights of each step.

