ssh datarmor
micromamba create -p /home1/datawork/todaka/micromamba/pangeo-fish-iaocea python=3.11.* jupyter intake-xarray zstandard git cf_xarray h5netcdf hvplot datashader pint-xarray flox dask-image movingpandas
micromamba activate /home1/datawork/todaka/micromamba/pangeo-fish-iaocea 
pip install dask-hpcconfig
cd /home1/datawork/todaka/git/github-iaocea
git clone git@github.com:IAOCEA/pangeo-fish.git
cd pangeo-fish
pip install -e .
ipython kernel install --name "pangeo-fish-iaocea" --user


