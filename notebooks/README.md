# Current demos

- [Base demo](pangeo-fish-base-demo.ipynb): temperature and acoustic evidence,
  movement fitting, state probabilities, trajectories and visualization.
- [Complete demo](pangeo-fish-complete-demo.ipynb): adds bathymetry, tide evidence
  and multiple movement parameters.

Both demos use public tag A19124 by default. Copernicus Marine credentials are
required for reference ocean-model data. Run each notebook from top to bottom.
The defaults shared by the demos are identical except for their output folders:

| Setting                              | Both demos                       |
| ------------------------------------ | -------------------------------- |
| BBOX                                 | Latitude 46–51.5, longitude −7–0 |
| HEALPix refinement level             | 10 (approximately 6.4 km)        |
| Optimization tolerance               | `1e-6`                           |
| Main trajectories                    | Viterbi, mode, mean              |
| Manual-sigma comparison trajectories | mode, mean                       |

In the Complete demo's **Study settings**, `tide_method = "healpix"` (default)
compares the shape of the tide with the depth record, with a model error
`tide_sigma_m`. `tide_method = "pedersen"` uses the tide likelihood of
[Pedersen et al. (2008)](https://doi.org/10.1139/f08-144)
(_Can. J. Fish. Aquat. Sci._ 65:2367), ported from the
[DTU / HMM geolocation MATLAB code](https://github.com/cliu3/hmm_smast/blob/dev/datalik/datalikelihood.m);
its error parameters are not calibrated yet. Both predict tides from local
harmonic constants directly on the HMM grid: install `.[notebooks,tide]`. See the
[tide methods guide](../docs/tide.rst) for when to use each.

The Complete demo also includes bathymetry and a two-sigma model. The
manual-sigma appendix uses a separate `comparison_track_modes` setting in both
demos; Viterbi with the appendix's large manually chosen sigma can require very
large transition tables.

## Run with Papermill

Install the optional notebook tools in the same environment as your kernel.
The Complete demo also needs pyfes: install it from conda-forge first (PyPI has
no wheels, so `pip install -e '.[notebooks,tide]'` compiles it).

```bash
conda install -c conda-forge pyfes
python -m pip install -e '.[notebooks]'
mkdir -p executed
papermill notebooks/pangeo-fish-base-demo.ipynb executed/base-A19124.ipynb \
  -k python3 -p tag_name A19124 -p scratch_root ./results/base
papermill notebooks/pangeo-fish-complete-demo.ipynb executed/complete-A19124.ipynb \
  -k python3 -p tag_name A19124 -p scratch_root ./results/complete
```

To compare with the `pedersen` tide method, use a separate results folder.

```bash
papermill notebooks/pangeo-fish-complete-demo.ipynb executed/complete-pedersen-A19124.ipynb \
  -k python3 -p tide_method pedersen -p scratch_root ./results/complete-pedersen
```

The `parameters` cell contains serializable defaults. Papermill inserts overrides
immediately after this cell; subsequent cells derive quantities, output paths
and chunks from those values. Supply speeds as `maximum_speed_km_per_day` and
receiver distances as `receiver_buffer_m`. To override structured values such as
`bbox`, use a YAML file with Papermill's `-f parameters.yaml` option. See the
[Papermill parameterization guide](https://papermill.readthedocs.io/en/latest/usage-parameterize.html).
Changing the study area or resolution changes the scientific experiment; keep
the common defaults when comparing tags.

The public long-duration A18828 case is restricted to base-demo in CI:

```bash
papermill notebooks/pangeo-fish-base-demo.ipynb executed/base-A18828.ipynb \
  -k python3 -p tag_name A18828 -p scratch_root ./results/base
```

It uses the same settings and the entire available tag period. The initial
GitHub Actions trial is manual (`dataset_case: A18828-base`), with a 45-minute
execution-step limit and a 55-minute job limit, so runtime can be measured before
adding it to automatic checks. A19124 runs both demos. Per-cell timings, effective
settings, source hashes and partial notebooks are retained, including on failure.
CI uses the pinned image in `ci/notebook-image.txt` and installs with dependencies
using `pip install -e '.[notebooks]'` before executing through Papermill.

Configure Copernicus Marine authentication in your environment before running;
never pass credentials as notebook parameters or commit them in parameter files.
The demo workflow uses the `DEMO_COPERNICUS_CREDENTIALS` Actions secret. It runs
the A19124 demos (about 5 minutes) on pushes and on pull requests to `main` and
`DEV` that touch `pangeo_fish/` or the demo files. In this repository, pushes to
branches other than `main` and `DEV` are checked through their pull request only.
A new push cancels the older run of the same pull request or branch, except on
`main` and `DEV`, where the running check finishes (GitHub keeps only the newest
waiting run, so quick successive merges can skip an intermediate commit).
GitHub does not pass secrets to pull requests from forks: there the demo is
skipped with a warning and the job still succeeds, so a green check does not mean
the notebooks ran. Such changes are checked by the push run in the fork (it needs
Actions enabled and its own `DEMO_COPERNICUS_CREDENTIALS` secret) or by running
the workflow by hand.

The [archive](archive/README.md) contains historical experiments, not supported
demos for this release.
