[![DOI](https://zenodo.org/badge/653040548.svg)](https://doi.org/10.5281/zenodo.15110142)
[![CI](https://github.com/pangeo-fish/pangeo-fish/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/pangeo-fish/pangeo-fish/actions/workflows/ci.yml)
[![docs](https://readthedocs.org/projects/pangeo-fish/badge/?version=latest)](https://pangeo-fish.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/pangeo-fish.svg)](https://pypi.org/project/pangeo-fish)
[![Downloads](https://pepy.tech/badge/pangeo-fish)](https://pepy.tech/project/pangeo-fish)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License](https://img.shields.io/github/license/pangeo-fish/pangeo-fish)](https://github.com/pangeo-fish/pangeo-fish/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/pangeo-fish/pangeo-fish)](https://github.com/pangeo-fish/pangeo-fish/commits/main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pangeo-fish/pangeo-fish/HEAD)

# Pangeo-Fish 🐟

**Pangeo-Fish** is an open, community-driven project designed to leverage the power of the Pangeo ecosystem specifically for fish biologging data analysis and related oceanographic research. The project facilitates scalable, reproducible, and collaborative research by integrating biologging data with environmental datasets using cloud-native technologies.

## 🌊 Project Goals

- **Simplify Biologging Data Analysis:** Integrate and streamline analyses of diverse fish biologging datasets with oceanographic and environmental data.
- **Leverage Cloud Infrastructure:** Utilize Pangeo technologies such as Xarray, Dask, and Jupyter to perform large-scale, efficient computations.
- **Promote Collaboration:** Foster an inclusive community where researchers and developers can share expertise, resources, and innovative tools.

## 🚀 Quick Start

Install directly from PyPI:

```bash
pip install pangeo-fish
```

Or set up the development environment:

```bash
git clone https://github.com/pangeo-fish/pangeo-fish.git
cd pangeo-fish
conda env create -n pangeo-fish -f environment.yml
conda activate pangeo-fish
jupyter lab
```

Explore example notebooks located in the [`notebooks`](https://github.com/pangeo-fish/pangeo-fish/tree/main/notebooks) directory.

## 📖 Documentation

Complete documentation, including guides, tutorials, and examples, is available on [Read the Docs](https://pangeo-fish.readthedocs.io/en/latest/).

Topics include:

- Data ingestion and preprocessing
- Integration of biologging and environmental datasets
- Scalable computing with Dask and Xarray
- Visualization techniques for biologging data

## 🤝 Contributing

We warmly welcome contributions! You can contribute by:

- Reporting bugs and suggesting enhancements via [GitHub Issues](https://github.com/pangeo-fish/pangeo-fish/issues).
- Joining discussions on our [GitHub Discussions](https://github.com/pangeo-fish/pangeo-fish/discussions).
- Submitting code improvements or new features through Pull Requests.

For detailed guidelines, please refer to our [contribution guidelines](https://github.com/pangeo-fish/pangeo-fish/blob/main/CONTRIBUTING.md).

## 📜 License

Pangeo-Fish is distributed under the [BSD 3-Clause License](https://github.com/pangeo-fish/pangeo-fish/blob/main/LICENSE).

---

🌐 **Join us to advance open, scalable science through biologging data analysis!**

## Acknowledgements

- T Odaka, JM Delouis and J Magin are supported by the CNES Appel, a projet R&T R-S23/DU-0002-025-01.
- T Odaka, JM Delouis and M Woillez are supported by the TAOS project funded by the IFREMER via the AMII OCEAN 2100 programme.
- Q Mazouni, M Woillez, A Fouilloux and T Odaka are supported by Global Fish Tracking System (GFTS), a Destination Earth use case procured and funded by ESA (SA Contract No. 4000140320/23/I-NS, DESTINATION EARTH USE CASES - DESP USE CASES - ROUND (YEAR 2023))
- M Woillez and T Odaka are supported by Digital Twin of the Ocean - Animal Tracking (DTO Track), a European project that aims to create a digital twin of the North Sea using animal tracking data and funded by the Sustainable Blue Economy Partnership (ANR-24-SBEP-0001)
