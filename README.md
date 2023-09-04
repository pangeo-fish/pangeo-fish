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
conda env create -n pangeo-fish -f ci/requirements/environment.yaml
conda activate pangeo-fish
```

(use the drop-in replacement `mamba` or `micromamba` for faster results)

Install the pangeo-fish downloaded above using pip by following command.

```sh

pip install -e .
```

## How to run the code

After the installation activate the envoriment pangeo-fish
Place the configuration file in a directory 'A18832_na'

type following command if you use the code on datarmor

Make sure to specify the absolute path to the enviomennt you have installed for executing the pangeo-fish after --enviroment path

./bin/run-workflow.sh --configuration-root ./configuration A18832_na --environment /home1/datawork/todaka/micromamba/envs/pangeo-fish-0723 --conda-path /appli/anaconda/versions/4.8.2/condabin/conda --memory "120GB" --walltime "4:00:00"
