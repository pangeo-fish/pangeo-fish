# Choosing a resolution: data size and memory

The HEALPix refinement level and the size of the domain set the number of
cells; together with the number of time steps they set the memory needed by the
whole workflow. Estimate it before choosing a level, especially for wide-ranging
species.

## Cell size per level

| Level | Cell area | Approximate cell size |
| ----- | --------- | --------------------- |
| 8     | 649 km²   | 25 km                 |
| 9     | 162 km²   | 13 km                 |
| 10    | 41 km²    | 6.4 km                |
| 11    | 10 km²    | 3.2 km                |
| 12    | 2.5 km²   | 1.6 km                |

Each level up multiplies the number of cells and the data size by four. The
number of cells of a domain is about its ocean area divided by the cell area.

## Size of the probability maps

The emission and state probabilities are stored as 64-bit floats: one map uses
`8 × n_cells` bytes, and a variable over the whole track uses
`8 × n_cells × n_time` bytes.

The fitting and state computations load the whole emission array into memory,
and the forward–backward filter keeps several arrays of the same size (the
predictions and states of both passes while they are stacked). The emission
arrays alone reach about **eight times the emission size** in `compute_states`
and about twice in `optimize_pdf`; this is a lower bound, without the
convolution kernel (below) nor Python and Dask overheads. The Viterbi decoding
additionally builds transition tables whose size grows with the movement
parameter.

The table below uses the area of each bounding box, land included, so it is an
upper bound; the ocean fraction lowers it.

| Domain                                              | Level | Cells  | One map | Emission, 365 steps | Emission arrays (×8) |
| --------------------------------------------------- | ----- | ------ | ------- | ------------------- | -------------------- |
| Demo box (7°W–0°, 46–51.5°N)                        | 10    | 8 k    | 62 kB   | 23 MB               | 0.2 GB               |
| Demo box (7°W–0°, 46–51.5°N)                        | 12    | 124 k  | 1 MB    | 0.4 GB              | 2.9 GB               |
| Mediterranean (6°W–36°E, 30–46°N)                   | 10    | 161 k  | 1.3 MB  | 0.5 GB              | 3.8 GB               |
| Mediterranean (6°W–36°E, 30–46°N)                   | 12    | 2.6 M  | 21 MB   | 7.5 GB              | 60 GB                |
| Mediterranean + North Atlantic (80°W–36°E, 10–65°N) | 9     | 371 k  | 3 MB    | 1.1 GB              | 8.7 GB               |
| Mediterranean + North Atlantic (80°W–36°E, 10–65°N) | 10    | 1.5 M  | 12 MB   | 4.3 GB              | 35 GB                |
| Mediterranean + North Atlantic (80°W–36°E, 10–65°N) | 11    | 5.9 M  | 48 MB   | 17 GB               | 139 GB               |
| Mediterranean + North Atlantic (80°W–36°E, 10–65°N) | 12    | 23.8 M | 190 MB  | 69 GB               | 555 GB               |

With hourly instead of daily steps, multiply the last two columns by 24.

## Convolution kernel

The movement model convolves each map with a Gaussian of width sigma. Its cost
depends on the convolution method (`conv_method`):

- `"HealpixConv"` (`Gaussian1DHealpix`, the default of `optimize_pdf`) stores a
  sparse kernel with about `(truncate × sigma / cell size)²` entries per cell,
  24 bytes each. For the same sigma in km, it grows **sixteen times per level**
  (four times more cells, each with four times more neighbours), and so does
  the time of each prediction step. On the demo box with `truncate=4` and a
  sigma of about 20 km, it uses about 30 MB at level 10, 0.6 GB at level 11 and
  about 10 GB at level 12; with sigma near the 60 km/day bound, about 70 GB at
  level 12. The optimizer evaluates sigmas up to that bound.
- `"LargeHealpixConv"` (`UpDownGaussian1DHealpix`, used by the demo notebooks)
  convolves on a coarser grid with a small compact kernel, so its kernel stays
  small at fine levels. Prefer it from level 11 upwards.

## Rules of thumb

With daily steps and `"LargeHealpixConv"`: a coastal box like the demo one runs
at level 12 with a few GB of memory; a regional sea such as the Mediterranean
fits at level 10 or 11 (about 4–15 GB for the emission arrays); a basin-scale
track, such as a tuna moving between the Mediterranean and the North Atlantic,
needs level 9 or 10, and a finer level only fits on a large-memory machine.
Add a margin of a few GB for Python, Dask and the kernel. With `"HealpixConv"`,
add the kernel size above, which dominates at levels 11 and 12. Reduce the
bounding box to the area the fish can reach, and the number of time steps to
what the observations resolve, before raising the level.

## Storage chunks

The intermediate HEALPix stores (`diff-regridded.zarr`, `emission.zarr`,
`combined.zarr`, `states.zarr`, ...) are written through
{py:func}`pangeo_fish.storage.prepare_dggs_zarr`, which sizes the chunks from
the data, up to 64 MiB per chunk:

- when one map is smaller than that (all the regional cases above), all the
  cells stay in one chunk and several time steps are grouped, e.g. about 1,000
  daily maps of the demo box at level 10 in one chunk;
- when one map is larger (basin scale at level 12), each chunk holds one time
  step and the cells are split into the fewest nearly equal chunks that fit.

The computations do not depend on the chunking: element-wise steps such as the
temperature emission work on any chunks, and the filter loads whole maps. Open
the stores with `chunks={}` (one Dask chunk per storage chunk) and rechunk
afterwards if needed, e.g. `xr.open_dataset(..., chunks={}).chunk({"time": 1})`.
Requesting Dask chunks smaller than the storage chunks directly in
`open_dataset` makes each storage chunk be read several times.
