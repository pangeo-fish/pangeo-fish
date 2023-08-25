# Installation

## Local computer

This documentation explains how to install panego-fish and how to create your environnement to run the different notebooks.

In this section, we assume that you already have Wsl and Conda environnement set up on your windows computer. If not, please refer to this page : [pangeo on windows](https://gitlab.ifremer.fr/diam/Pangeo-on-Windows)

If you already have Conda ready, you are good to go.

First, clone the pangeo-fish repo :

```console
git clone https://github.com/IAOCEA/pangeo-fish.git
```

Then, create an environnement with the following command :

```console
micromamba create -n pangeo-fish python=3.11.* jupyterlab=3.* holoviews=1.16.2 intake-xarray zstandard git cf_xarray h5netcdf hvplot datashader pint-xarray flox dask-image movingpandas geoviews=1.10.0 cmocean geopandas cartopy ffmpeg dask-labextension jupyterlab-git papermill -y
```

This will create your environnement with all the required libraries to make the pangeo-fish lib work.

Since the library itself has not been published anywhere, so for now it has to be installed from source:

```console
cd pangeo-fish
pip install -e .
```

And install your environnement as a kernel :

```console
ipython kernel install --name "pangeo-fish" --user.
```


## Datarmor

You can refer to this documentation on how to use [pangeo on hpc](https://gitlab.ifremer.fr/diam/pangeo_on_HPC).

You can reproduce all of the steps above it's the same, you will jsut have to add  `dask-hpc-config` to work on HPC :

```console
pip install dask-hpcconfig
```

All of those steps should create a environnement that can be runned on the different pangeo-fish notebooks 