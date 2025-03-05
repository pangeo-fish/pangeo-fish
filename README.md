# pangeo-fish

## installing

The library itself has not been published anywhere, so for now it has to be installed from source:

```sh
git clone https://github.com/iaocea/pangeo-fish.git
cd pangeo-fish
```

The main dependencies are:

- xarray
- pandas
- numpy
- scipy
- numba
- more-itertools
- opt_einsum
- sparse
- healpy

Install them by creating a new `conda` environment:

```sh
mamba env create -n pangeo-fish -f ci/requirements/environment.yaml
conda activate pangeo-fish
```

(use the drop-in replacement `mamba` or `micromamba` for faster results)

Install the pangeo-fish downloaded above using pip by following command.

```sh

pip install -e .
```
