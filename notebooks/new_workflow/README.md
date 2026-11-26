# Notebooks — Pipeline guide

Full pipeline for light-based geolocation of archival DST tags.
Run the notebooks **in order** for each new tag.

---

## Tag data format

Every tag must be stored as a folder `{TAG_ROOT}/{tag_name}/` containing three files.

### `dst.csv`

Time series of the tag sensor, one row per record:

| column        | type                        | description             |
| ------------- | --------------------------- | ----------------------- |
| `time`        | ISO 8601 UTC string, index  | timestamp of the record |
| `temperature` | float (°C)                  | external temperature    |
| `pressure`    | float (dbar)                | depth / pressure        |
| `light`       | float (raw counts) or `NaN` | raw light intensity     |

**Requirements:**

- `time` must be **sorted** (monotone increasing) and **without duplicates** — `sel(time=slice(...))` will fail otherwise.
- Extra columns are allowed and ignored by `load_tag`, but only `temperature`, `pressure`, `light` are used by the pipeline.

### `tagging_events.csv` (or `tagging_event.csv`)

Two rows — release and end of deployment:

| column       | description                                |
| ------------ | ------------------------------------------ |
| `event_name` | `release` or `fish_death` (or `recapture`) |
| `time`       | ISO 8601 UTC string                        |
| `longitude`  | decimal degrees                            |
| `latitude`   | decimal degrees                            |

### `metadata.json`

Free JSON dict, stored as tag attributes. Minimum useful fields: `tag_name`, `tag_type`.

---

## Tag types and light channel

| `tag_type`        | Manufacturer / format                          | Light column                                                                                | `HAS_LIGHT`       |
| ----------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------- |
| `lotek`           | Lotek LAT2810 — semicolon-separated raw CSV    | `LightIntensity` → renamed to `light` by `prepare_tag_folder`                               | `True`            |
| `wc_psat`         | Wildlife Computers MiniPAT Series CSV          | no light sensor → `light = NaN`                                                             | **`False`**       |
| `dst`             | Already a standard `dst.csv`                   | depends on the source                                                                       | depends           |
| _(pre-formatted)_ | WC or other tag with light, manually converted | column name varies (e.g. `Light Level`) — must be renamed to `light` in Ch6 of the notebook | `True` if renamed |

**Wildlife Computers PSAT (`wc_psat`):** the `-Series.csv` file from Wildlife Computers only records depth and temperature at 10-minute intervals — no raw light counts. Set `HAS_LIGHT = False` to skip the solar and lunar chapters.

**WC tags with light (e.g. 18P0430, 20P0204):** these are pre-formatted manually. The light column may be named `Light Level` or `light` depending on how the CSV was exported. Ch6 of the notebook loads the column explicitly (`tag["/dst"].ds[["Light Level", ...]]`) and renames it to `light` for the solar pipeline. Set `HAS_LIGHT = True`.

---

## Step 0 — Convert raw manufacturer CSV to standard format

**Notebook:** `raw_to_dst.ipynb`
**Run:** once per tag, on your local machine
**Input:** raw CSV from the tag manufacturer
**Output:**

```
{TAG_ROOT}/{tag_name}/
    dst.csv            ← time, temperature, pressure, light
    tagging_events.csv ← release + fish_death events
    metadata.json      ← tag_name, tag_type
```

Uses `pangeo_fish.light.ingest.prepare_tag_folder(raw_csv_path, tag_type, ...)`.
Supported `tag_type`: `"lotek"`, `"wc_psat"`, `"dst"`.

> Skip this step if the tag folder already exists in `TAG_ROOT`.

---

## Step 1 — Download the CMEMS model

**Notebook:** `prepare_tag_model.ipynb`
**Run:** once per tag
**Requires:** Copernicus Marine credentials (`copernicusmarine` CLI logged in)
**Input:** formatted tag folder from Step 0
**Output:** `{target_root}/model.zarr` saved to S3

---

## Step 2 — Compute the temperature variance profile

**Notebook:** `variance_fast.ipynb`
**Run:** once per tag
**Requires:** `model.zarr` from Step 1, Argo data fetched from ERDDAP (cached locally after first run)
**Output:** `{target_root}/temperature_variance.zarr` saved to S3

---

## Step 3 — Run the main pipeline

**Notebook:** `pangeo_fish_with_light.ipynb`
**Run:** as many times as needed
**Requires:** Steps 1 and 2 completed
**Output:** emission PDFs, HMM trajectory, plots — all saved to `{target_root}/` on S3

Configure the tag at the top of the notebook (Chapter 1):

- `tag_name` — tag identifier
- `TAG_ROOT` — root folder or S3 path containing the formatted tag folder
- `HAS_LIGHT` — `True` if the tag has usable raw light counts (enables Ch6 solar + Ch7 lunar PDFs)
- `bbox` — spatial bounding box
- `scratch_root` / `storage_options` — S3 destination

---

## Other notebooks

| Notebook                       | Purpose                                |
| ------------------------------ | -------------------------------------- |
| `pannel_plot.ipynb`            | Diagnostic plots for a completed run   |
| `coastal_distance_study.ipynb` | Bathymetry / coastal distance analysis |
| `papermill/`                   | Batch execution via papermill          |

---

## Tag data on S3

Tags stored at `s3://gfts-ifremer/tuna/tags/formatted/`:

| Tag         | Type          | Light                          | Notes                       |
| ----------- | ------------- | ------------------------------ | --------------------------- |
| `281B-4949` | Lotek LAT2810 | `light` (raw counts)           | NW Mediterranean, 22 months |
| `18P0430`   | WC with light | `Light Level` → renamed in Ch6 | Mediterranean, ~1 year      |
| `20P0204`   | WC with light | `light` (pre-renamed)          | Atlantic, ~15 months        |
