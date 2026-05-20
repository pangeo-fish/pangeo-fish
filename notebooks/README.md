# Notebooks — Pipeline guide

Full pipeline for light-based geolocation of archival DST tags.
Run the notebooks **in order** for each new tag.

---

## Step 0 — Convert raw manufacturer CSV to standard format

**Notebook:** `raw_to_dst.ipynb`
**Run:** once per tag, on your local machine
**Input:** raw CSV from the tag manufacturer (Lotek, Wildlife Computers…)
**Output:**

```
{TAG_ROOT}/{tag_name}/
    dst.csv            ← time, temperature, pressure, light
    tagging_event.csv  ← release + fish_death events
    metadata.json      ← tag_name, tag_type, species…
```

> Skip this step if the tag is already formatted (files exist in `TAG_ROOT`).

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
- `TAG_TYPE` — `"lotek"` or `"wc_psat"`
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

## Tag data folders

Local tag folders (used as `tag_root` for offline runs):

| Folder       | Tag       | Notes                   |
| ------------ | --------- | ----------------------- |
| `20P0204/`   | 20P0204   | Lotek, local CSV format |
| `281B-4949/` | 281B-4949 | Lotek, NW Mediterranean |
