API Reference
=============


High-level Functions
--------------------


.. currentmodule:: pangeo_fish.helpers
.. autosummary::
   :toctree: generated

   to_healpix
   reshape_to_2d
   load_tag
   update_stations
   plot_tag
   load_model
   compute_diff
   open_diff_dataset
   regrid_dataset
   estimate_refinement_costs
   check_event_position
   compute_emission_pdf
   compute_acoustic_pdf
   combine_pdfs
   normalize_pdf
   optimize_pdf
   predict_positions
   plot_trajectories
   open_distributions
   plot_distributions
   render_frames
   render_distributions



I/O
---

.. currentmodule:: pangeo_fish.io
.. autosummary::
   :toctree: generated

   open_tag
   write_tag
   open_copernicus_catalog
   open_copernicus_zarr
   prepare_dataset
   save_trajectories
   read_trajectories
   save_html_hvplot
   tz_convert


ETN Archival Data
-----------------


ETN Acoustic Detections
-----------------------

.. currentmodule:: pangeo_fish.etn_acoustic
.. autosummary::
   :toctree: generated

   find_etn_acoustic_detections


DGGS Zarr Output
----------------

.. currentmodule:: pangeo_fish.etn_archival
.. autosummary::
   :toctree: generated

   open_etn_archival
   read_etn_animals


Grid Manipulation
-----------------

.. currentmodule:: pangeo_fish
.. autosummary::
   :toctree: generated

   grid.center_longitude
   healpy.geographic_to_astronomic
   healpy.astronomic_to_cartesian
   healpy.astronomic_to_cell_ids
   healpy.buffer_points


Emission Computations
---------------------

.. currentmodule:: pangeo_fish
.. autosummary::
   :toctree: generated

   acoustic.emission_probability
   pdf.normal
   pdf.combine_emission_pdf
   diff.diff_z


Tides on HEALPix
----------------

Tidal harmonic constants on HEALPix cells, tide prediction with nodal corrections
(requires ``pyfes``, extra ``pangeo-fish[tide]``) and the tidal likelihood of a tag
lying still on the seabed.

See :ref:`tide-methods` for the Complete demo selector and the comparison
with the legacy method.

.. currentmodule:: pangeo_fish.tide_healpix
.. autosummary::
   :toctree: generated

   make_cell_ids
   cell_centers
   harmonic_constants_dataset
   open_harmonic_constants
   select_constants
   as_complex
   write_harmonic_constants
   nodal_modulations
   predict_tide
   still_tag_tide_likelihood
   write_tide_pdf_healpix
   harmonic_analysis
   constants_misfit
   latlon_constants_from_tmd
   latlon_constants_from_amplitude_phase
   constants_to_healpix


Tag/time Operations
-------------------

.. currentmodule:: pangeo_fish
.. autosummary::
   :toctree: generated

   tags.to_time_slice
   tags.adapt_model_time
   tags.reshape_by_bins
   cf.bounds_to_bins
   dataset_utils.broadcast_variables


Visualization
-------------

.. currentmodule:: pangeo_fish.visualization
.. autosummary::
   :toctree: generated

   create_single_frame
   render_frame
   plot_map
   plot_trajectories
   filter_by_states


Estimators
----------

.. currentmodule:: pangeo_fish.hmm.estimator
.. autosummary::
   :toctree: generated

   EagerEstimator
   CachedEstimator


Predictors
----------

.. currentmodule:: pangeo_fish.hmm.prediction
.. autosummary::
   :toctree: generated

   Gaussian2DCartesian
   Gaussian1DHealpix


Searches
--------

.. currentmodule:: pangeo_fish.hmm.optimize.scipy
.. autosummary::
   :toctree: generated

   GridSearch
   EagerBoundsSearch
   TargetBoundsSearch


Low-level Functions
-------------------

Distributions
#############

.. currentmodule:: pangeo_fish.distributions
.. autosummary::
   :toctree: generated

   planar2d.create_covariances
   planar2d.normal_at
   healpix.normal_at
   healpix.angular_distances


Trajectory Generation
#####################

.. currentmodule:: pangeo_fish.hmm.decode
.. autosummary::
   :toctree: generated

   mean_track
   modal_track
   viterbi


HMM Filtering
#############

.. currentmodule:: pangeo_fish.hmm.filter
.. autosummary::
   :toctree: generated

   score
   forward
   backward
   forward_backward


Dry Cells
#########

.. currentmodule:: pangeo_fish.drycells
.. autosummary::
   :toctree: generated

   dry_mask
   count_dry_cells
   mask_dry_cells
   DryCellWarning


ETN Acoustic Detections
-----------------------

.. currentmodule:: pangeo_fish.etn_acoustic
.. autosummary::
   :toctree: generated

   find_etn_acoustic_detections
   ETNAcousticDetections


