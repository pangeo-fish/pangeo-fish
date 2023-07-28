# pangeo-fish

## installing

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
conda env create -n pangeo-fish -f ci/requirements/environment.yaml
conda activate pangeo-fish
```

(use the drop-in replacement `mamba` for faster results)

The library itself has not been published anywhere, so for now it has to be installed from source:

```sh
git clone https://github.com/iaocea/pangeo-fish.git
cd pangeo-fish
pip install -e .
```
