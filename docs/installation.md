# Installation

This documentation explains how to install panego-fish and how to create your environnement to run the different notebooks.

In this section, we assume that you already have Wsl and Conda environnement set up on your windows computer. If not, please refer to this page : [pangeo on windows](https://gitlab.ifremer.fr/diam/Pangeo-on-Windows)

If you already have Conda ready, you are good to go.

First, clone the pangeo-fish repo :

```console
git clone https://github.com/IAOCEA/pangeo-fish.git
```

Then, create an environnement with the following command :

```console
micromamba create -n pangeo-fish -f ci/requirements/environment.yaml
```

This will create your environnement with all the required libraries to make the pangeo-fish lib work.

Since the library itself has not been published anywhere, so for now it has to be installed from source:

```console
cd pangeo-fish
pip install -e .
```

You will also need `dask-hpcconfig` for your environnement :

```console
pip install dask-hpcconfig
```

And install your environnement as a kernel :

```console
ipython kernel install --name "pangeo-fish" --user.
```

You can refer to this documentation on how to use [pangeo on hpc](https://gitlab.ifremer.fr/diam/pangeo_on_HPC) such as datarmor.

All of those steps should help create a environnement that can be runned on the different pangeo-fish notebooks
