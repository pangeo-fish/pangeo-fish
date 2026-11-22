# Choosing the movement model (`conv_method`)

In each simulation time step the fish can move. `pangeo-fish` models this move as a Gaussian step. The width of this step is `sigma`. The convolution (`conv_method`) is the code that applies this step to the probability map.

Pick it in `optimize_pdf(..., conv_method="...")`. If you do nothing, you get `FFTConv`.

## Sigma in kilometres

Think of sigma as the distance a fish moves in one time step, in km. The methods have limits in cells, so you need both units.

One cell is `sqrt(pi/3) / 2**level` radians wide (the convention used in the checks). With the Earth radius of 6371 km this gives:

| HEALPix level | Cell size (km) |
| ------------- | -------------- |
| 8             | 25.5           |
| 9             | 12.7           |
| 10            | 6.37           |
| 11            | 3.18           |
| 12            | 1.59           |

So at level 10, sigma = 4 cells is about 25 km per step.

## Quick choice

| Your case                            | Use                                       |
| ------------------------------------ | ----------------------------------------- |
| Most runs (default)                  | `FFTConv`, `projection="conservative"`    |
| Fish that move little per step       | `HealpixAnalyseConv`                      |
| Islands, headlands, narrow channels  | `HealpixAnalyseConv`, `land_barrier=True` |
| Domain larger than 10 degrees radius | `HealpixAnalyseLargeConv`                 |

## 1. Supported by measurements

All numbers below come from the demo tag A19124 at level 10. Sigma was tested from 1 to 32 cells (about 6 to 204 km per step). The "cost" is the value the optimiser minimises.

### Default: `FFTConv` with `projection="conservative"`

Use this unless you have a reason not to. Why:

- It follows the direct sum closely. Its cost differs from the direct sum by less than 1 % at every sigma where both were run (1, 2, 3, 4, 6 and 8 cells). Across a face boundary, the largest difference to the direct sum is 2.8e-3, for a direct peak of 1.76e-2.
- It is smooth in sigma (see below).
- It is fast: 0.07 to 0.46 s per cost evaluation, and about 800 MB of memory, for every sigma tested.

Limit: the domain must fit in a radius of 10 degrees (one tangent plane). A larger domain stops with this error: "FFTConv cannot handle this domain: its radius is X degrees, above the limit of 10 degrees". The Japan box in the checks needs 15.5 degrees and stops this way.

### Fish that move little per step

`HealpixAnalyseConv` computes the direct sum. It is exact, but its cost grows fast with sigma. Measured (default `normalization`, no barrier):

| Sigma (cells) | km (level 10) | Time (s) | Memory       |
| ------------- | ------------- | -------- | ------------ |
| 1             | 6.4           | 3.29     | not measured |
| 2             | 13            | 10.97    | not measured |
| 3             | 19            | 20.87    | not measured |
| 4             | 25            | 34.71    | not measured |
| 6             | 38            | 82.2     | 13.2 GB      |
| 8             | 51            | 141.8    | 18.9 GB      |

The times for 1 to 4 cells were taken while other jobs were running, so read them as rough values.

From sigma = 6 cells on, the operator has many millions of entries. Above 8 cells the checks skipped it (operator over 5e7 non-zeros, at 12 cells and above about 3e8). Use it for small steps only.

### Islands, headlands, narrow channels

Use `HealpixAnalyseConv` with `land_barrier=True`. The kernel is then measured through water and no probability crosses land. `FFTConv` and `HealpixAnalyseLargeConv` cannot do this. The barrier makes it slower: with `normalization="source"` it took 6.23, 24.19, 63.1 and 131.57 s at sigma 1, 2, 3 and 4 cells (6 to 25 km). Larger sigma was skipped (same size limit).

### Domain larger than 10 degrees radius

Use `HealpixAnalyseLargeConv`. It has no domain limit, but it has a minimum sigma between 2 and 3 cells (13 to 19 km at level 10; the healpix-analyse documentation gives about 2.2 cells, and the exact limit depends on the patch). Below that it raises an error and suggests `HealpixAnalyseConv`. In the checks it failed at 1 and 2 cells and ran from 3 cells on. It runs in 0.28 to 5.5 s.

It deviates from the direct sum. At 8 cells (51 km) its cost is 36.20, against 32.82 for the direct sum, a difference of about 10 %. At 6 cells it is 35.72 against 35.21. The direct sum was not run for larger sigma, so the deviation there is not known.

### Sigma optimisation and smoothness

The optimiser looks for the sigma with the lowest cost. This works only if the cost changes smoothly with sigma. The methods that go through coarser levels (`LargeHealpixConv` and `HealpixAnalyseLargeConv`) do not: their cost moves in steps or bumps. The previous up/down implementation gave the identical cost 32.18293 at sigma = 2 and 3 cells, and 33.53 and 33.54 at 4 and 6 cells. The optimiser can stop at a wrong value on such a plateau. The current versions are better but still leave the smooth curve, for example at 8 cells.

`FFTConv` (conservative) and the direct sum change smoothly and stay together:

| Sigma (cells)                     | 1     | 2     | 3     | 4     | 6     | 8     | 12      | 16      | 24      | 32      |
| --------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ------- | ------- | ------- | ------- |
| `FFTConv`                         | 26.89 | 28.55 | 30.19 | 31.90 | 35.25 | 32.56 | 28.06   | 24.96   | 22.77   | 22.99   |
| `HealpixAnalyseConv` (direct sum) | 26.87 | 28.33 | 30.03 | 31.78 | 35.21 | 32.82 | not run | not run | not run | not run |
| `HealpixAnalyseLargeConv`         | error | error | 32.12 | 33.05 | 35.72 | 36.20 | 29.94   | 25.21   | 21.52   | 20.50   |
| `LargeHealpixConv`                | 74.57 | 36.34 | 32.09 | 33.01 | 35.92 | 35.04 | 29.00   | 24.53   | 21.11   | 21.17   |

Prefer `FFTConv` when the optimiser must find sigma.

### Mass lost on land

With the default `normalization="full_disc"`, probability that falls on land or leaves the domain is lost. The loss enters the cost. With `normalization="source"` (`HealpixAnalyseConv` only) the total is kept. Choose it if you want the loss to stay out of the cost.

For `FFTConv`, `projection="conservative"` keeps the total at 1.00000 in the boundary check. `projection="average"` gave 1.04133 there (0.99422 in the control), so the total drifts. This is why conservative is the default.

## 2. Not yet verified

- Long-distance migrants. `FFTConv` stops at 10 degrees radius. The accuracy of `HealpixAnalyseLargeConv` was measured only up to the sigma above. Splitting the domain into smaller patches is not implemented.
- Islands combined with large steps. The direct sum runs out of memory, and no method does both today.

## 3. In development

Fish that enter rivers narrower than one cell need a mixed-level operator (`multiorder_filter_operator` in `healpix-analyse`). It is not yet connected to `pangeo-fish`.

## Last word

How well small regional temperature differences pin down the position depends on the observation (emission) model and the reference model, not on the convolution. See [Choosing a Copernicus Marine ocean model](copernicus-models.md) for the reference model.
