# HEALPix Zarr output and Gridlook Preview

HEALPix datasets saved through the workflow helpers and the maintained demos
carry the [DGGS Zarr v1 convention](https://github.com/zarr-conventions/dggs/blob/v1/README.md).
The group's `dggs` attribute describes the level, indexing scheme, spatial
dimension, cell coordinate and numeric reference ellipsoid. `zarr_conventions`
identifies the versioned convention. The `cell_ids` coordinate also keeps the
attributes needed by xdggs 0.6, so it can still be opened with `ds.dggs.decode()`.

`prepare_dggs_zarr(ds)` prepares a dataset for a new store: cells last, one time
step per chunk and up to 4096 cells per spatial chunk. It preserves the sparse
cell domain and order used for inference. It clears inherited storage encodings
before rechunking, so source chunk boundaries and zero-valued fill sentinels do
not leak into the new store. Non-storage variable attributes are preserved.
This helper does not write a store. For example:

```python
from pangeo_fish.storage import prepare_dggs_zarr

prepare_dggs_zarr(emission).to_zarr(
    "combined.zarr", mode="w-", zarr_format=2, consolidated=True
)
```

## Export parent-cell chunks

For viewing, `write_dggs_zarr` uses the dense parent-cell layout used by
[GRID4EARTH healpix-convert](https://grid4earth.eu/healpix-convert/): each spatial
chunk contains all the children of one occupied parent cell. This requires
fixed-level **nested** indexing. Simply splitting a sparse list every 4096
entries would not align the chunks with HEALPix parents.

```python
from pangeo_fish.storage import write_dggs_zarr

# states is the Dataset returned by compute_states(). A fresh path is required.
write_dggs_zarr(states, "states-preview.zarr")
```

The default parent level is `max(0, data_level - 6)`: at most 4096 children per
chunk. For example, level 10 data uses level 4 parents. Set `chunk_level=5` to
use 1024 children per chunk instead. Each occupied parent is padded separately;
unoccupied parents are omitted. Cell IDs are sorted, zero IDs and zero
probabilities are preserved, and added cells have NaN values (False for boolean
masks). No interpolation or probability normalization is performed. Integer
variables with added missing values may become floating point. The input dataset
is unchanged; use the original sparse data for inference.

This is a single-resolution export in **Zarr format 2**, with Blosc/LZ4 and
consolidated metadata, supported by the Gridlook Preview reader. It does not
require Zarr v3, sharding, custom codecs or a multiscale pyramid. The DGGS
convention and Zarr storage version are separate choices. Dense padding increases
the number of stored cells; choose a finer parent level for very sparse data.

## Write to a bucket and open over HTTPS

```python
# Use the endpoint and credentials for your writable bucket, such as the
# bucket supplied by your Pangeo@EOSC deployment. Do not put secrets in notebooks.
write_dggs_zarr(
    states,
    "s3://YOUR-BUCKET/YOUR-PREFIX/states-preview.zarr",
    storage_options=storage_options,
)
```

The S3 write URL and the HTTPS read URL can differ. Obtain the HTTPS URL from
your storage provider. Bucket membership alone does not establish public read
access or browser access. Before opening Gridlook, check:

- HTTPS GET can read `.zgroup`, `.zattrs`, `.zmetadata`, the array metadata
  (`states/.zarray`, `cell_ids/.zarray`) and their chunk objects.
- The response permits the viewer's origin `https://grid4earth.eu` through CORS.
  Allow `GET` and `HEAD`; allow the `Range` request header if requested by the
  reader. CORS also needs to apply to missing-object responses used in discovery.
- The viewer can read without your notebook's S3 credentials. For restricted
  data, use an access mechanism supported by your viewer and storage deployment.

Open the **store root**, for example:

```text
https://grid4earth.eu/gridlook-preview/#https://YOUR-ENDPOINT/YOUR-PREFIX/states-preview.zarr
```

Choose `states` in the variable selector and move the time selector. The
[Gridlook Preview](https://grid4earth.eu/gridlook-preview/) is an experimental
viewer deployment. Exporting a store does not publish it or change bucket
permissions. A new S3 endpoint still needs a live browser check; local round-trip
tests cannot verify its read policy, HTTPS routing or CORS configuration.

## Published A19124 example

A real-data interoperability example is available in the Pangeo@EOSC MinIO
bucket `todaka-pangeo-fish-demo`, exported with commit `0609f50`:

[Open A19124 in Gridlook Preview](https://grid4earth.eu/gridlook-preview/#https://pangeo-eosc-minioapi.vm.fedcloud.eu/todaka-pangeo-fish-demo/outputs/A19124/dggs-preview/2026-11-09-0609f50/states.zarr)

Select `states`. The dataset contains eleven daily posterior maps from
2022-06-14 to 2022-06-24, recomputed from the existing workshop's A19124 combined
temperature/acoustic emission and its saved movement sigma
(`0.0014121326945640952`, spherical grid). It is a visualization interoperability
demonstration rather than a newly validated tracking result.

The export uses level-10 nested cells and level-4 parent tiles: chunks of
`(time=1, cells=4096)`, 81,920 exported cells including padding, and 39,388
original cells. It preserves the source posterior values and each daily map
sums to one. The Zarr store occupies about 67 kB across 51 objects (all-NaN
chunks are omitted). Existing bucket policy and CORS responses were sufficient;
neither was changed.

The complete dataset has been reopened anonymously over HTTPS and compared
with the local export. Metadata and populated chunks return the viewer origin
in their CORS headers. See the adjacent
[validation record](https://pangeo-eosc-minioapi.vm.fedcloud.eu/todaka-pangeo-fish-demo/outputs/A19124/dggs-preview/2026-11-09-0609f50/validation.json)
for the export details. An HTTP round trip alone does not verify the browser's
rendering of the maps.
