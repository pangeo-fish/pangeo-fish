.. _tide-methods:

Tidal evidence in the Complete demo
===================================

Choose ``tide_method`` in **Study settings** at the top of the
`Complete demo notebook
<https://github.com/pangeo-fish/pangeo-fish/blob/main/notebooks/pangeo-fish-complete-demo.ipynb>`_.

.. code-block:: python

    tide_method = "legacy"  # Original method (default)
    # tide_method = "healpix"  # Local harmonic prediction on the HMM grid

Both methods use the existing depth-based flags. TPXO9 is the default tide
model. Section 7 downloads it from Ifremer, and ``tide_model_source`` can
point to a local copy of the same TMD3 NetCDF file. Install the optional tide
dependency in the notebook kernel's environment to use the HEALPix method.

.. code-block:: console

   python -m pip install -e '.[notebooks,tide]'

Using FES2022b
--------------

Obtain ``FES2022b_OceanTide_NSgrid.nc`` from AVISO+ and select these values in
the Complete demo's Study settings.

.. code-block:: python

    tide_method = "healpix"
    tide_model = "fes2022b"
    tide_model_source = "/path/to/FES2022b_OceanTide_NSgrid.nc"
    ellipsoid = "WGS84"

PyFES applies the native six-node LGP2 interpolation to the complex
constituents at the HMM cell centres. Cells outside the native
mesh have missing constants. For example, the native mesh reaches the lower
Scheldt estuary near Bath, while the release site at Temse is outside its
coverage. The demo uses the ocean tide elevation constituents from the file.

See the `AVISO FES product page
<https://www.aviso.altimetry.fr/index.php?id=1279>`_ and
the `AVISO data license
<https://aviso.altimetry.fr/en/data/data-access/fileadmin/documents/data/License_Aviso.pdf>`_
for access and attribution.

Reading precomputed FES2022b constants
-------------------------------------

A precomputed store contains the real and imaginary parts of the harmonic
constants in metres at nested WGS84 HEALPix cell centres. It must cover every
candidate cell at the run's HEALPix level. Cells outside FES2022b's native mesh
remain in the store with missing values.

Start the kernel with AWS credentials configured for the bucket. For a named
profile, set ``AWS_PROFILE`` to its name in the environment before starting
Jupyter or Papermill. Replace the example URI and endpoint with your own values
and set the HEALPix level to match your store.

.. code-block:: python

    tide_method = "healpix"
    tide_model = "fes2022b"
    ellipsoid = "WGS84"
    refinement_level = 10
    tide_model_source = None
    tide_constants_store = "s3://<bucket>/<prefix>/ocean-tide-constants.zarr"
    tide_s3_endpoint = "https://s3.example.org"

The option also accepts a local Zarr path. Use ``tide_constituents`` to select
a list of waves on either FES2022b path, or leave it as ``None`` for all waves.
The level, ellipsoid, indexing scheme and candidate cell IDs must match.
To return to the local native NetCDF, clear ``tide_constants_store`` and set
``tide_model_source`` as above.

To prepare a store for a different study grid, pass the candidate cell IDs
with their DGGS attributes to ``fes2022b_constants_to_healpix`` and write the
result with ``write_harmonic_constants``. Set ``chunk_constituents=1`` for
independent reads by constituent. Include the AVISO citation, licence,
product version and interpolation method in the store metadata.

How the likelihoods differ
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Step
     - ``legacy``
     - ``healpix``
   * - Tide prediction
     - Builds a tide signal from the file's constituent amplitudes and phases.
       This time-dependent signal is shared across the model grid.
     - Obtains complex harmonics at the HMM cell centres by interpolation
       or by reading precomputed FES2022b constants. PyFES supplies the
       astronomical arguments and nodal corrections at the observation times.
   * - Depth reference
     - Compares observed depth to the model water-column depth adjusted by the
       shared signal. Shallow-water cells can be favoured by absolute depth.
     - Fits an unknown mean depth per cell and per day. The tide term uses the
       shape of the depth record during candidate near-bed intervals. The demo
       supplies absolute depth evidence through its separate bathymetry PDF.
   * - Time aggregation
     - Produces hourly likelihoods on the latitude/longitude grid, regrids
       them to HEALPix, then averages them by day. A day labelled ``T`` covers
       ``[T - 12 h, T + 12 h)``, as in the temperature emission.
     - Averages observations and tide predictions over the same sample times
       within each hour, then computes one likelihood per day on the HMM grid
       (day ``T`` covers ``[T - 12 h, T + 12 h)``, as in the temperature emission).
       ``tide_sigma_m`` sets the model error and ``tide_min_samples`` sets
       the minimum number of valid hourly means for a day to contribute.

The HEALPix likelihood assumes the tag is lying still on the seabed during the
selected intervals. The demo uses the tide-detection flag to identify
candidate intervals. :func:`pangeo_fish.tide.detect_tide` slides a window of
25 h (about two M2 cycles; never shorter than 12.42 h) along the pressure record.
In each window it removes a linear trend, fits an M2 harmonic, and flags the
window when the R² on the detrended series is above 0.85, the residual RMS is
below 0.42 m and the amplitude lies between 0.6 and 5 m. Each flag is stored at
the centre of its window, so the first and last 12.5 h have no flag. Ascents,
descents and diel vertical migration are not flagged. The ``behav_found``
variable is the same fit with relaxed thresholds, not a separate behaviour
classifier. Check the assumption of a still tag when adapting the method to
another tag. Days with too few valid hourly means get uniform weight over the cells
with usable harmonic constants.

Comparing the outputs
---------------------

Set a different ``scratch_root`` for each run. Section 7 displays the selected
method and its daily tide PDF. Both methods write ``tide_pdf_healpix.zarr``
with a ``pdf`` variable. Section 8 writes the combined temperature, acoustic,
bathymetry and tide PDF to ``combined.zarr``.

Match both ``time`` and ``cell_ids`` when comparing runs. Renormalize over
the common wet cells if you want to compare how the two methods distribute
probability over the same area. Comparing the combined PDFs shows how much of
that difference remains after including the other evidence.
