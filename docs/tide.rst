.. _tide-methods:

Tidal evidence in the Complete demo
===================================

Choose ``tide_method`` in **Study settings** at the top of the
`Complete demo notebook
<https://github.com/pangeo-fish/pangeo-fish/blob/main/notebooks/pangeo-fish-complete-demo.ipynb>`_.

.. code-block:: python

   tide_method = "legacy"   # Original method (default)
   # tide_method = "healpix"  # Local harmonic prediction on the HMM grid

Both methods use the demo's TPXO9 model and the existing depth-based flags.
The ``tide_model_source`` setting also accepts a local path to the same TMD3
NetCDF file. Install the optional tide dependency in the notebook kernel's
environment to use the HEALPix method.

.. code-block:: console

   python -m pip install -e '.[notebooks,tide]'

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
     - Interpolates the local complex harmonic constants to the HMM cell
       centres, then predicts a different tide at each cell. PyFES supplies the
       astronomical arguments and nodal corrections at the observation times.
   * - Depth reference
     - Compares observed depth to the model water-column depth adjusted by the
       shared signal. Shallow-water cells can be favoured by absolute depth.
     - Fits an unknown mean depth per cell and per day. The tide term uses the
       shape of the depth record during candidate near-bed intervals. The demo
       supplies absolute depth evidence through its separate bathymetry PDF.
   * - Time aggregation
     - Produces hourly likelihoods on the latitude/longitude grid, regrids
       them to HEALPix, then averages them by day.
     - Averages observations and tide predictions over the same sample times
       within each hour, then computes one likelihood per day on the HMM grid.
       ``tide_sigma_m`` sets the model error and ``tide_min_samples`` sets
       the minimum number of valid hourly means for a day to contribute.

The HEALPix likelihood assumes the tag is lying still on the seabed during the
selected intervals. The demo uses the existing tide-detection flag to identify
candidate intervals. Check that assumption when adapting the method to another
tag. Days with too few valid hourly means get uniform weight over the cells
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
