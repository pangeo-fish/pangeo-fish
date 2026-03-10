# Installation

This documentation covers the installation process of `pangeo-fish` in a virtual Python environment.

In the following, we expect Conda to be installed on your Linux-based system.

For Windows users, we recommend WSL (2.0).

For more information, please refer to this page: [pangeo on Windows](https://gitlab.ifremer.fr/diam/Pangeo-on-Windows).

## For first use

If you are only interested in using `pangeo-fish` locally (i.e, without accessing remote data nor HPC resources), then all is needed is to create you create a virtual environment and install the package as well as its depedencies.

To do so, clone the `pangeo-fish`'s repository and navigate to it:

```console
git clone https://github.com/pangeo-fish/pangeo-fish.git
cd pangeo-fish
```

Then, create a conda or micromamba environment with the following command:
Still, we recommend using micromamba since it’s faster. Use **either** of the following commands:

```console
# Using conda
conda env create -n pangeo-fish -f docs/environment.yaml

# Or using micromamba
micromamba create -n pangeo-fish -f docs/environment.yaml
```

And then just activate the environment

```console
conda/micromamba activate pangeo-fish
```

This will create your environment with all the required libraries to make `pangeo-fish` work.

Finally, install the package itself from either its repository:

```console
pip install -e .
```

... or from `pip`:

```console
pip install pangeo-fish
```

## For HPC use

This section details the additional steps to setup `pangeo-fish` for HPC use.

As such, we assume the reader has access to HPC resources such as [Datarmor](https://www.ifremer.fr/fr/infrastructures-de-recherche/le-supercalculateur-datarmor).

_Please ensure to have completed the installation introduced above first!_

Install the `dask-hpcconfig` package and set the environnement as a Jupyter kernel:

```console
pip install dask-hpcconfig
ipython kernel install --name "pangeo-fish" --user
```

Besides, you can refer to [this documentation on how to use pangeo on HPC](https://dask-hpcconfig.readthedocs.io/en/latest/).

All of those steps should create a Python environment able to run the different Jupyter notebooks available.

## Help and Troubleshooting

In case of errors or difficulties upon installing `pangeo-fish`, you can report the issue [here](https://github.com/pangeo-fish/pangeo-fish/issues).
