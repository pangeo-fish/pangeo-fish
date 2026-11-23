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

This will create your environment with all the required libraries to make `pangeo-fish` work, including the CPU-only build of PyTorch (`pytorch-cpu`).

Finally, install the package itself from either its repository:

```console
pip install -e .
```

... or from `pip`:

```console
pip install pangeo-fish
```

## PyTorch: CPU or GPU

`pangeo-fish` depends on PyTorch, and runs it on the CPU by default (`device="cpu"`).
On Linux, `pip` takes the CUDA build of PyTorch from PyPI, which adds the `nvidia-*` and `triton` packages (about 3 GB).
If you are not using a GPU, install the CPU-only build **before** `pangeo-fish`; `pip` then sees that PyTorch is already installed and does not replace it:

```console
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The conda environment above already contains the CPU-only build (`pytorch-cpu` from conda-forge).

If you want to run on a GPU, install the CUDA build instead:
`pip install torch` (the default build on Linux; see the [PyTorch selector](https://pytorch.org/get-started/locally/) for the build that matches your CUDA version), or, with conda, replace `pytorch-cpu` by `pytorch-gpu` in `docs/environment.yaml`.

From a GPU allocation, check that PyTorch can see a CUDA device:

```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

The second value should be `True`. In the workflow, pass `device="cuda"` to both `optimize_pdf` and `compute_states` and select a CUDA-capable convolution method in `optimize_pdf`. The default method, `FFTConv`, supports CUDA. See [Choosing a convolution method](choosing-conv-method.md#running-on-a-gpu-device) for the full list. The demo notebooks use the CPU by default, and `decode_trajectories` runs on the CPU.

Total runtime includes predictor setup, CPU work and transfers between host and GPU memory. Compare CPU and GPU times for your dataset if runtime is a concern.

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
