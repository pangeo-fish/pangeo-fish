# Choosing a Copernicus Marine ocean model

`pangeo-fish` compares the temperatures recorded by a tag with the temperatures
of an ocean model. When no local model is given, the model is read directly from
the [Copernicus Marine Service](https://marine.copernicus.eu/) (a free account is
needed; run `copernicusmarine login` once).

## How to select a model

The model is chosen with the `copernicus_model_name` argument of
{py:func}`pangeo_fish.helpers.load_model`, together with `uri=None`. Pass the
**temperature** `dataset_id`; the matching sea surface height (`zos`) and
bathymetry (`deptho`, `mask`) datasets are found automatically.

```python
from pangeo_fish.helpers import load_model

reference_model = load_model(
    uri=None,  # read from Copernicus Marine
    copernicus_model_name="cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m",
    tag_log=tag_log,
    time_slice=time_slice,
    bbox=bbox | {"max_depth": tag_log["pressure"].max()},
)
```

To switch to another model, change only the `copernicus_model_name` string,
for example to `"cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i"` for the global
6-hourly model. In the demo notebooks this string is set in the configuration
cell (Section 2), where the alternatives are listed as commented-out lines.

The result always has the same variables (`TEMP`, `XE`, `H0`), so the rest of
the workflow does not change. For sub-daily models, the time step of the
analysis follows the model (hourly or 6-hourly instead of daily), which
increases the cost of the later steps.

## Supported models

Time coverage was checked with `copernicusmarine describe` on **2026-11-19**.
Analysis/forecast products are extended every day; the regional hourly 3D
datasets are a **rolling window of about two years**: old data are deleted, so
their start date moves forward over time.

| Product                                                                                                                                 | `copernicus_model_name` (temperature)           | Grid                                 | Time step                         | 3D  | Coverage (2026-11-19)             | Sea surface height from                                                    | Tides in `zos`       |
| --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------ | --------------------------------- | --- | --------------------------------- | -------------------------------------------------------------------------- | -------------------- |
| [IBI reanalysis](https://data.marine.copernicus.eu/product/IBI_MULTIYEAR_PHY_005_002/description) (base demo default)                   | `cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m`      | 1/36° (~3 km), Iberia–Biscay–Ireland | daily mean (`P1D-m`)              | yes | 1993-01-01 → 2026-08-25           | `cmems_mod_ibi_phy-ssh_my_0.027deg_P1D-m`                                  | averaged out (daily) |
| [IBI analysis/forecast](https://data.marine.copernicus.eu/product/IBI_ANALYSISFORECAST_PHY_005_001/description)                         | `cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m`     | 1/36° (~3 km), Iberia–Biscay–Ireland | hourly mean (`PT1H-m`)            | yes | 2024-11-25 → 2026-11-28 (rolling) | `cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m`                                | yes                  |
| [North-West Shelf analysis/forecast](https://data.marine.copernicus.eu/product/NWSHELF_ANALYSISFORECAST_PHY_004_013/description)        | `cmems_mod_nws_phy-tem_anfc_1.5km-3D_PT1H-i`    | 1.5 km, North-West European Shelf    | hourly instantaneous (`PT1H-i`)   | yes | 2024-11-15 → 2026-11-26 (rolling) | `cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT1H-i`                               | yes                  |
| [GLORYS12 global reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) (`load_model` default) | `cmems_mod_glo_phy_my_0.083deg_P1D-m`           | 1/12° (~9 km), global                | daily mean (`P1D-m`)              | yes | 1993-01-01 → 2026-10-06           | same dataset                                                               | averaged out (daily) |
| [Global analysis/forecast](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description)                   | `cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m`  | 1/12° (~9 km), global                | daily mean (`P1D-m`)              | yes | 2022-06-01 → 2026-11-28           | `cmems_mod_glo_phy_anfc_0.083deg_P1D-m`                                    | averaged out (daily) |
| [Global analysis/forecast](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description)                   | `cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i` | 1/12° (~9 km), global                | 6-hourly instantaneous (`PT6H-i`) | yes | 2022-06-01 → 2026-11-29           | `cmems_mod_glo_phy_anfc_0.083deg_PT1H-m`, sampled at the temperature times | no                   |

### Which model to use

- **First run with the base demo:** IBI reanalysis. Long record, regional resolution.
- **Outside the IBI area:** GLORYS12 (before mid-2022) or the global analysis/forecast.
- **Sub-daily analysis:**
  - use the IBI or North-West Shelf hourly models when their rolling coverage
    includes the tag dates and study area.
  - use the global 6-hourly model for tags from 2022-06-01 outside those windows.

### Notes and caveats

- **Requested period not fully covered:** if the model does not cover the whole
  requested period (for example a rolling hourly dataset that no longer goes back
  far enough), the model is cut to the period it covers and a `UserWarning` is
  issued.
- **Global 6-hourly model:**
  - There is no 6-hourly sea surface height. `zos` is read from the hourly-mean
    2D dataset (stamped on the hour) and sampled at 00, 06, 12 and 18 UTC.
    At the end of the forecast, use the period covered by both temperature and
    `zos`. If this clips the data, the loader issues a `UserWarning`.
  - The hourly-mean dataset `cmems_mod_glo_phy_anfc_0.083deg_PT1H-m` also has
    `thetao`, but at the surface only (one level at 0.49 m). It is therefore not
    a 3D temperature and is not offered as a `copernicus_model_name`.
- **Global `zos` has no tides:**
  - Near the Scheldt mouth (3.0° E, 51.6° N), the hourly `zos` varies by only
    0.1–0.4 m over two days, against about 2.9 m in the IBI and North-West Shelf
    hourly `zos`.
  - The depth correction (`XE`) is therefore not tidal with the global models.
- **Sea surface height datum:**
  - `zos` is not on the same reference level in all products. In the southern
    North Sea, the global analysis/forecast was about 15 cm lower on average than
    the IBI reanalysis (May–June 2023).
  - At one point in May 2025, the two-day means were −0.26 m (global), −0.25 m
    (IBI hourly) and −0.20 m (North-West Shelf hourly).
  - This does not matter within one analysis (a single model is used), but do
    not mix `zos` from different products.
- **Bathymetry on land:** the North-West Shelf static dataset stores a depth
  value (`deptho`) on land cells as well. `pangeo-fish` sets `deptho` to NaN
  wherever `mask` marks land, for every product.
- **Model differences:** in the same area, the daily-mean surface temperatures
  of the models differ by a few tenths of a degree:
  - global 6-hourly (averaged per day) vs global daily: within 0.05 °C (2023);
  - global vs IBI reanalysis: +0.14 °C on average (2023);
  - global vs North-West Shelf daily: −0.38 °C on average (2025).

  Keep this in mind when choosing the temperature tolerance (`differences_std`).
