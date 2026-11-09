# Historical notebooks

These files are retained as historical experiments. They are excluded from the
current release's supported demos and notebook execution tests. They have not
been validated against the current Pangeo Notebook environment.

## Legacy Papermill and coastal-distance experiments

- `papermill/pangeo-fish_papermill.ipynb` has a Papermill `parameters` cell,
  historical BARGIP/S3 settings and resource-reporting instrumentation.
- `papermill/pangeo-fish_coastal_distance.ipynb` is a related experiment with
  extra visualization of the coastal adjustment and older estimator APIs.
- `papermill/resource_reporter.py` is preserved with the notebook that used it.

Both notebooks apply a coastal-distance adjustment: they divide the temperature
residual by `1 + exp(-distance_scale_factor * distance**2)` before constructing
the temperature likelihood. This changes the scientific model; it is not part
of adding Papermill execution support.

A future coastal-distance demo should start from the current maintained demos
and document distance units, the rationale for the adjustment and its scale,
and comparisons with and without the correction. Merely updating the legacy
APIs would not establish that the correction is scientifically appropriate.

Use the [current demos](../README.md) for the maintained workflow.
