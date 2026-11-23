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
convolution (below) nor Python and Dask overheads. The Viterbi decoding
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

## Convolution

The movement model convolves each map with a Gaussian of width sigma. Its
memory and time depend on the method (`conv_method`); see
[Choosing the movement model](choosing-conv-method.md) for the measurements:

- `"FFTConv"` (the default) used about 800 MB per cost evaluation on the demo
  tag at level 10, for every sigma tested. It only accepts domains within a
  radius of 10 degrees: the demo box fits, the whole Mediterranean (more
  than 15 degrees from its centre to its ends) does not.
- `"HealpixConv"` and `"HealpixAnalyseConv"` (direct sums) store an operator
  whose number of entries per cell grows as `(sigma / cell size)²`. For the same
  sigma in km it grows **sixteen times per level** (four times more cells, each
  with four times more neighbours), and so does the time of each step: 13 GB
  at level 10 for a sigma of 38 km on the demo tag.
- `"HealpixAnalyseLargeConv"` has no domain limit, which makes it the option
  for basin-scale tracks, within its minimum and maximum sigma.

## Rules of thumb

With daily steps: a coastal box like the demo one runs at level 12 with a few
GB for the emission arrays and `"FFTConv"`. A regional sea such as the
Mediterranean fits at level 10 or 11 (about 4–15 GB for the emission arrays);
a basin-scale track, such as a tuna moving between the Mediterranean and the
North Atlantic, needs level 9 or 10, and a finer level only fits on a
large-memory machine. Both exceed the `"FFTConv"` radius and need
`"HealpixAnalyseLargeConv"`. Add the convolution and a margin of a few GB for
Python and Dask.
Reduce the bounding box to the area the fish can reach, and the number of time
steps to what the observations resolve, before raising the level.

## Storage chunks

The intermediate HEALPix stores (`diff-regridded.zarr`, `emission.zarr`,
`combined.zarr`, `states.zarr`, ...) are written through
{py:func}`pangeo_fish.storage.write_healpix_zarr`, which sizes the chunks from
the data, up to 64 MiB per chunk:

- when one map is smaller than that (all the regional cases above), all the
  cells stay in one chunk and several time steps are grouped, e.g. about 1,000
  daily maps of the demo box at level 10 in one chunk;
- when one map is larger (basin scale at level 12), each chunk holds one time
  step and the cells are split into the fewest nearly equal chunks that fit.

The computations do not depend on the chunking: element-wise steps such as the
temperature emission work on any chunks, and the filter loads whole maps. Open
the stores with {py:func}`pangeo_fish.storage.open_healpix_zarr`, which reads
each storage chunk once and then rechunks, e.g.
`open_healpix_zarr(path, chunks={"time": 1, "cells": -1})`. Requesting Dask
chunks smaller than the storage chunks directly in `xr.open_dataset` makes each
storage chunk be read several times.