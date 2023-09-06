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

### Interactive way
You can run the code from jupyter-lab interface using jupyternotebooks existing in `./notebooks/workflows`.  You can updates parameters in the first cells


### Non interactive way
If you want to run the notebooks automatically, by providing parameters in configuration files.  
To do so, place parameters in `congiguration\A19124` directory.  You will find examples there `01_copernicus_diff.yaml`

Place parameter files in `yaml` format for each notebooks you will want to run sequentially.  pangeo-fish can submit your job to batchsytem, one after another, in alphaverical order.

Example of command lines for Datarmor:
```
ssh datarmor
bash
micromamba activate pangeo-fish-0723
# cd to pangeo-fish installed directory
cd /home1/datawork/todaka/git/pangeo-fish

./bin/run-workflow.sh --configuration-root ./configuration A19124 --environment /home1/datawork/todaka/micromamba/envs/pangeo-fish-0723 --conda-path /appli/anaconda/versions/4.8.2/condabin/conda --memory "120GB" --walltime "4:00:00"

```

Make sure to specify the absolute path to the enviomennt you have installed for executing the pangeo-fish after --enviroment path

