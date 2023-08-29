```
ssh datarmor
bash
micromamba create -p /home1/datawork/todaka/micromamba/envs/pangeo-fish-0723 python=3.11.* jupyterlab=3.* holoviews=1.16.2 intake-xarray zstandard git cf_xarray h5netcdf hvplot datashader pint-xarray flox dask-image movingpandas geoviews=1.10.0 cmocean geopandas cartopy ffmpeg dask-labextension jupyterlab-git papermill -y
micromamba activate /home1/datawork/todaka/micromamba/envs/pangeo-fish-0723
pip install dask-hpcconfig
cd /home1/datawork/todaka/git/github-iaocea
git clone git@github.com:IAOCEA/pangeo-fish.git
cd pangeo-fish
pip install -e .
ipython kernel install --name "pangeo-fish-0723" --user
#testing now
# This works except 08_track_decoding

```
