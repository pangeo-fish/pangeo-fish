.. _tide-methods:

Tidal evidence in the Complete demo
===================================

Choose ``tide_method`` in **Study settings** at the top of the
`Complete demo notebook
<https://github.com/pangeo-fish/pangeo-fish/blob/main/notebooks/pangeo-fish-complete-demo.ipynb>`_.

.. code-block:: python

    tide_method = "healpix"  # Tide shape on each cell, model error tide_sigma_m (default)
    # tide_method = "pedersen"  # Pedersen et al. (2008) tide likelihood

Both methods predict the tide on the HMM cells from local harmonic constants and
use the depth-based tide flags of :func:`pangeo_fish.tide.detect_tide`. TPXO9 is
the default tide model. Section 7 downloads it from Ifremer, and
``tide_model_source`` can point to a local copy of the same TMD3 NetCDF file.
Install the optional tide dependency in the notebook kernel's environment.

.. code-block:: console

   python -m pip install -e '.[notebooks,tide]'

Using FES2022b
--------------

Obtain ``FES2022b_OceanTide_NSgrid.nc`` from AVISO+ and select these values in
the Complete demo's Study settings.

.. code-block:: python

    tide_method = "healpix"  # or "pedersen"
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

    tide_method = "healpix"  # or "pedersen"
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

The ``healpix`` likelihood
--------------------------

:func:`pangeo_fish.tide_healpix.still_tag_tide_likelihood` obtains the complex
harmonic constants at the HMM cell centres by interpolation or by reading
precomputed FES2022b constants. PyFES supplies the astronomical arguments and
nodal corrections at the observation times. The observations and the tide
predictions are averaged over the same sample times within each hour, and one
likelihood per day is computed on the HMM grid. A day labelled ``T`` covers
``[T - 12 h, T + 12 h)``, as in the temperature emission.

The likelihood fits an unknown mean depth per cell and per day, so it uses the
shape of the tide in the depth record and not the absolute depth. The demo
supplies absolute depth evidence through its separate bathymetry PDF. The errors
are independent with the standard deviation ``tide_sigma_m`` (the error of the
tide model, surge and sensor, not the variability of the depth within an hour).
``tide_min_samples`` sets the minimum number of valid hourly means for a day to
contribute; a day with fewer gets a uniform tide PDF.

The ``pedersen`` likelihood
---------------------------

``tide_method = "pedersen"`` is a port of the tide likelihood of the HMM
geolocation toolbox of DTU [Pedersen2008]_. The MATLAB source is
in ``datalik/datalikelihood.m`` and ``datalik/datalikparam.m`` of
`cliu3/hmm_smast <https://github.com/cliu3/hmm_smast/blob/dev/datalik/datalikelihood.m>`_
(ver. 0.52, 2008). The implementation is
:func:`pangeo_fish.tide_healpix.pedersen_tide_likelihood`; its docstring lists
every departure from the MATLAB.

For each window of the depth record in which the tide detection finds a tide,
the depth samples ``y`` are one Gaussian vector around the depth predicted at
cell ``c``, ``mu_c(t) = D_c + eta_c(t)``. ``D_c`` is the mean water depth of the
cell from GEBCO (positive down, referenced to mean sea level) and ``eta_c`` the
tide elevation from the harmonic constants of the cell with the nodal
corrections of pyfes. The covariance does not depend on the cell: white noise
(0.2 m), a correlated term with the rmse of the tide fit of the window and a
memory of 6.67 h, and a constant 10 m standard deviation on the mean depth.
A day labelled ``T`` combines the windows whose centre lies in
``[T - 12 h, T + 12 h)``. A day without a tidal window gets a uniform tide PDF:
the bathymetry PDF of the demo already says that the fish cannot be where the
seabed is shallower than its deepest depth. ``non_tidal="bathymetry"`` restores
the bathymetry-only likelihood of the MATLAB, which then counts the bathymetry
twice.

The error parameters are those of the MATLAB code and have not been calibrated
for this package:

- the 10 m standard deviation on the mean depth (``depth_offset_sd``) makes the
  absolute depth almost uninformative inside a tidal window, as in the MATLAB;
- the correlated error uses the rmse of the window fit of ``detect_tide``, which
  the detection thresholds keep below 0.42 m;
- ``bathy_variance`` (only used with ``non_tidal="bathymetry"``) defaults to
  the 20 m\ :sup:`2` debug value of the MATLAB source;
- ``D_c`` assumes that GEBCO refers to mean sea level, which may not hold near
  the coast.

Comparing the two methods
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - ``healpix`` (default)
     - ``pedersen``
   * - Data used
     - Hourly means of the depth and of the predicted tide.
     - Raw samples of the flagged windows (one 25 h window per day).
   * - Mean depth
     - Unknown, fitted for every cell and day. Absolute depth comes from the
       bathymetry PDF only.
     - Mean water depth of the cell from GEBCO, with a 10 m standard deviation:
       weak absolute depth evidence.
   * - Errors
     - Independent, standard deviation ``tide_sigma_m``.
     - Time-correlated (white noise, AR(1) with a 6.67 h memory, constant
       offset). One Cholesky factor per window, log space.
   * - Days without a flagged window
     - Uniform.
     - Uniform (``non_tidal="bathymetry"``: the MATLAB bathymetry-only
       likelihood).
   * - Origin
     - Written for this package.
     - Published method [Pedersen2008]_, ported from its MATLAB code.
   * - Known weaknesses
     - Biased towards cells with a small tidal amplitude when the fish is not
       lying still on the bottom (the observed amplitude is then smaller than
       the predicted one).
     - Sensitive to the datum of the bathymetry; error parameters not
       calibrated. Memory grows with (samples per window) x (cells).

Which method should I use?
--------------------------

- Use ``healpix`` (the default) when the bathymetry is uncertain, for example
  near the coast or in estuaries: it does not depend on the datum of the
  bathymetry, because it fits the mean depth.
- Use ``pedersen`` when you want results that can be compared with the published
  method or with the MATLAB toolbox, and the bathymetry is reliable.
- Neither method has been validated against a tag with a known track. When you
  rely on the tide evidence, run both and compare them (see
  `Comparing the outputs`_).
- Both methods rely on ``detect_tide`` to find the windows in which the fish is
  lying still on the bottom. For a fish that does not do so, no window is
  flagged, the tide PDF is uniform on every day, and the choice of method makes
  no difference. Check ``tide_found`` first.

Tide detection
--------------

:func:`pangeo_fish.tide.detect_tide` slides a window of 25 h (about two M2
cycles; never shorter than 12.42 h) along the pressure record. In each window it
removes a linear trend, fits an M2 harmonic, and flags the window when the R² on
the detrended series is above 0.85, the residual RMS is below 0.42 m and the
amplitude lies between 0.6 and 5 m. The thresholds are those of the MATLAB
``tidebehavextr.m`` [Pedersen2008]_. Each flag is stored at the centre of its
window, so the first and last 12.5 h have no flag. Gaps in the record are kept
as gaps, so a window always spans 25 h, and a window with fewer than 80 % of its
samples is not flagged. The ``behav_found`` variable is the same fit with
relaxed thresholds, not a separate behaviour classifier.

Limits of the detection:

- **Micro-tidal areas are not detected.** The amplitude must be at least 0.6 m,
  so an M2 amplitude below that (for example in the Baltic or the
  Mediterranean) gives no flag.
- **Only a fish lying still on the bottom is flagged.** In synthetic records,
  a still fish under M2 and S2 is flagged in every window, whereas a linear
  ascent, a sine or square diel migration and noise are never flagged. These
  tests do not cover real tags: a fish that follows the tide in the water
  column, or rarely touches the bottom, may get no flagged window.
- **Results of earlier versions differ.** The detection window was 10 h
  (16 h for ``behav_found``) and flags were stored at the start of the window,
  and the tide PDF of day ``T`` used ``[T, T + 24 h)``. The window is now 25 h,
  flags are centred, and the day is ``[T - 12 h, T + 12 h)`` like the
  temperature emission, so a run made before this change is not reproduced.

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

Checks that fail loudly
-----------------------

The tide helpers stop or warn instead of silently changing the evidence:

- ``merge_tide_with_total`` keeps every cell and day of the total emission.
  Cells without harmonic constants (or, for ``pedersen``, without a water
  depth), cells absent from the tide grid and days absent from the tide store
  get a neutral constant factor, and so do days without a flagged window
  (``1 / n_cells``). They are unknown, not impossible. The neutral factor of a
  day with only some cells missing is the mean of its available tide values.
  A warning reports the counts. ``initial``, ``final`` and ``mask`` are not
  modified.
- ``convert_lon_360_to_180`` raises ``ValueError`` when the ``bbox`` lies outside
  the tide file (the demo TPXO subset only covers the North Sea and the
  Channel), and warns with the clipped extent when it is only partly covered.
- The pressure record must be in dbar or m: ``prepare_pressure`` (and so
  ``detect_tide``) raises if the ``units`` attribute of the variable says
  otherwise (Pa, hPa, mbar, ...). Both ``prepare_pressure`` and
  ``tide_behav_extr`` raise if the values are impossible for a depth (above
  3000 or below -50). ``tide_behav_extr`` only sees a ``units`` attribute when
  it is set on the DataFrame passed to it (``df.attrs["units"]``). A record
  without a ``units`` attribute in a unit of a similar magnitude (for example
  kPa) is not detected.
- ``tide_sigma_m`` is only used by ``tide_method = "healpix"``. With
  ``"pedersen"`` the demo warns (via ``check_tide_sigma``) when it differs from
  the default of 2.0, which that method does not read.

References
----------

.. [Pedersen2008] Pedersen, M. W., Righton, D., Thygesen, U. H., Andersen, K. H.
   and Madsen, H. (2008). Geolocation of North Sea cod (*Gadus morhua*) using
   hidden Markov models and behavioural switching. *Can. J. Fish. Aquat. Sci.*
   65, 2367-2377. `doi:10.1139/f08-144 <https://doi.org/10.1139/f08-144>`_
