---
title: "pangeo-fish: A Python package for studying fish movement using biologging and earth science data"
tags:
  - Python
  - Pangeo
  - Dask
  - Xarray
  - Kerchunk
  - Biologging
  - Geolocation
  - Earth Science
authors:
  - name: Justus Magin
    orcid: 0000-0002-4254-8002
    affiliation: 1
  - name: Quentin Mazouni
    orcid: 0009-0003-3519-5514
    affiliation: 2
  - name: Etienne Cap
    orcid: 0009-0007-0360-0692
    affiliation: 1
  - name: Marine Gonse
    orcid: 0000-0002-5378-8482
    affiliation: 3
  - name: Mathieu Woillez
    orcid: 0000-0002-1032-2105
    affiliation: 3
  - name: Anne Fouilloux
    orcid: 0000-0002-1784-2920
    affiliation: 2
  - name: Jean-Marc Delouis
    orcid: 0000-0002-0713-1658
    affiliation: 1
  - name: Tina Odaka
    orcid: 0000-0002-1500-0156
    affiliation: 1
affiliations:
  - name: LOPS (Laboratory for Ocean Physics and Satellite remote sensing) UMR 6523, Univ Brest-Ifremer-CNRS-IRD, Plouzané, France
    index: 1
  - name: Simula Research Laboratory, Oslo, Norway
    index: 2
  - name: DECOD (Ecosystem Dynamics and Sustainability), IFREMER-Institut Agro-INRAE, Plouzané, France
    index: 3
date: 05 March 2025
bibliography: paper.bib
---

# Summary

<!-- Geo-referenced data plays an important role in understanding and conserving natural resources, particularly when investigating biological phenomena such as fish migration and habitat uses.
Biologging, the practice of attaching small devices (called _tags_) to animals for recording behavior, physiology, and environmental data, proves to be invaluable in this field. -->

As fish can not be tracked directly using tracking devices such as GPS receivers, geolocation models have emerged to estimate fish positions by correlating individual series of physical measurements — e.g. temperature and pressure records — with geophysical reference fields — oceanic temperature and bathymetry — derived from satellite observations and hydrodynamical model outputs.

Beside the difficulty of working with vast earth science datasets (due to their size and diversity), there is no open source implementation for biologged fish tracking.
Yet, these fish geolocation models are critical for better understanding fish behavior and are nowadays seen as a powerful tool by policy makers to improve fish management and conservation.

To address this challenge, we developed a Python package, named **pangeo-fish**, for <u>fish tracking estimation</u>.
It is based on the [Pangeo](https://www.pangeo.io/) ecosystem, which offers a unique interoperable, scalable, open source environment for interactive data analysis in the field of big data marine and geoscience.

# Statement of need

Biologging consists in attaching onto (or sometimes inserting into) an animal an
electronic device that will record in its memory physical and/or geochemical parameters as a function of time so that scientists can reconstruct the activity of the animal, the characteristics of the environment it travels in and the interactions between the two.

These tools can provide a wealth of information on the behaviors and movements of free-swimming marine animals, such as diving and activity patterns, energy use and interaction with environment.

![Promotion of the FISH-INTEL tagging campaign.\label{fig:fishintel}](fishintel.png){ width=35% }

However, unlike terrestrial animals or marine mammals, whose positions can be directly estimated using ARGOS or GPS technologies, tracking fish underwater is challenging.
To address this issue, various tagging experiments have been conducted on a variety of fish species [@spanish_tagging; @skagerrak_tagging], and methods have been proposed for approximating the fish locations, referred to as geolocation models [@pontual_seabass_migration_2023; @woillez_hmm-based_2016].

For studying fish movements, the two widely used electronic tagging technologies are acoustic telemetry and Data Storage Tags (DST, or archival tags).
Acoustic telemetry involves a tag that emits an acoustic signal containing a unique ID and possibly sensor data.
This signal can be detected by an acoustic receiver when the tagged animal is within range, and the detection data is retrieved from the receiver.
As such, acoustic tags do not need to be recovered, but there is no guarantee that the tagged fish will swim around the receivers network.

In contrast, archival tags store sensor measurements at set intervals in their memory.
To access the logged data, these tags must either be recovered (which mostly depends on fishers and the local population living along the coast) or transmit their information via satellite.
In the former case, tagging campaigns usually promote and possibly reward tag or fish captures (see for instance, the advertisement from the [FISH-INTEL](https://www.france-energies-marines.org/en/projects/fish-intel/) campaign on \autoref{fig:fishintel}).
The data from archival tags can offer detailed insights into vertical movement patterns [@heerah2017coupling] or environmental preferences [@righton2010thermal; @skagerrak_tagging], and can be used to reconstruct migration paths through geolocation modeling.

![Example of an acoustic tag (on the left) and a DST (on the right). See the centimeter scale for size reference.\label{fig:tag}](archival_tag.png){ width=35% }

\autoref{fig:tag} shows an example of an acoustic tag as well as a DST.

The estimation of fish positions depends on the likelihood of the observed data from the DTS's logs, such as temperature at specific depths, alongside the reference geoscience data such as satellite observations and ocean dynamic models.
Some approaches can enhance the accuracy of the model's predictions by using additional information, such as telemetric detection data from the acoustic tags mentioned above [@a_combination_tag_2023].
The use of oceanic models with high spatial and temporal resolutions can significantly improve the accuracy of reconstructed fish tracks.
However, higher resolutions involves more data, that requires significant computing power and storage capacity.
The [Pangeo community](https://www.pangeo.io/) handles these challenges, by fostering an ecosystem of interoperable, scalable, open source tools for interactive data analysis in the field of big data marine and geoscience.
Therefore, the Pangeo ecosystem represents a powerful mean through which biologists can analyze more easily their biologging data and improve fish geolocation modelling.
Not only their results would eventually guide policy makers to manage fish stock in a more sustainable way, they could also be used to forecast potential movement changes due to the ongoing climate change.

Unfortunately, the research community lacks of adaptable, scale and open source implementations of geolocation models.
**pangeo-fish** is a Python package that fills this gap.

As its name suggest, the software has been designed to be used within the Pangeo ecosystem on several aspects, therefore accounting for both the users' needs (user-friendly API and meaningful result visualization) and the computational challenges.
In particular, **pangeo-fish** has a robust data model based on [Xarray](https://docs.xarray.dev/en/latest/index.html) and scales computation with [Dask](https://docs.dask.org/en/stable/).

Data loading processes are furthermore streamlined by libraries like `intake`, `kerchunk` or `fsspec`, and the previously mentioned `xarray` data model enables interactive visualization of the results thanks to tools such as the `hvplot` library and the JupyterLab environment.

Similarly, `pangeo-fish`'s I/O operations are automatically distributed with the combination of Dask and [Zarr](https://zarr.dev/).
The Pangeo software stack provides researchers with the necessary tools to access reference data and perform intensive computations in a scalable and interactive manner.
`pangeo-fish` gives ecologists a user-friendly tool for inferring fish locations from archival tag data, hence filling the gap between their expertise and the Pangeo's environment capabilities.

# Geolocation Model

**pangeo-fish** implements a method well established in the fish trajectory reconstruction literature [@pedersen2008geolocation, @woillez_hmm-based_2016, @a_combination_tag_2023].
It consists of a Hidden Markov Model (HMM).

![Illustration of the Hidden Markov Model. The hidden states $Xt$ describe the fish's positions, and the emission probabilities $P(Y_t|X_t)$ correspond to the likelihood of observing the fish at time $t$.\label{fig:hmm}](hmm.png){ width=75% }

As illustrated in \autoref{fig:hmm}, the latent (or _hidden_) states $X_t$ of the HMM infer the (daily or hourly) fish's positions, and the observation process relates the sensor records with the oceanic data.
The transition matrix between the hidden states is modelled by a Brownian motion parametrized by $\sigma$.
As such, fitting the geolocation model for a tag's records aims to determine the value of $\sigma$ that maximizes the likelihood of the state sequence (i.e., the fish's trajectory) given the observations.
The optimal likelihood value reflects the level of residual inconsistency between the tag observed (recorded) and reference data.

# Key Features of pangeo-fish

**pangeo-fish** is a Python software that handles the entire pipeline for reconstructing fish trajectories given sensor records and a reference data, and visualizing the results.
The key features of **pangeo-fish** are its simple API that follows each stage of the pipeline, and its scalability, which includes parallel computation and streamlined remote data fetching.

## Pre-processing

The framework starts by loading the archival records of a tagged fish:

```python
from pangeo_fish.helpers import load_tag

tag, tag_log, time_slice = load_tag(tag_root, tag_name, storage_options)
```

Then, the user selects the reference model (depending on, for example, the studied area or the resolution of the model):

```python
from pangeo_fish.helpers import load_model

reference_model = load_model(uri, tag_log, time_slice, ...)
```

In the previous instruction, the oceanic model is reduced to fit the scope of the tag (in time and space) `load_model()`.

The next step involves complex operations, which are however made easily accessible for the user in a few instructions.
The operations consists of computing the difference between the archival tag and reference data, reshaping the consequent result (to avoid spatial distortions) to finally compute the emission probabilities given the initial (and optionally the final) position(s) of the fish and the reshaped dataset:

```python
from pangeo_fish.helpers import compute_diff, regrid_dataset, compute_emission_pdf

diff = compute_diff(reference_model, tag_log, ...)[0]
reshaped = regrid_dataset(diff, ...)[0]
emission_pdf = compute_emission_pdf(differences, tag["tagging_events"].ds, ...)[0]
```

## Accounting for acoustic detections

Before normalizing the emission probabilities, the user can include another distribution, based on the possible fish detections by acoustic receivers:

```python
from pangeo_fish.helpers import compute_acoustic_pdf, combine_pdfs

acoustic_pdf = compute_acoustic_pdf(emission_pdf, tag, receiver_buffer, ...)
combined_pdf = combine_pdfs([emission_pdf, acoustic_pdf], ...)
```

## Model optimization

Finally, the normalized distributions are fitted to find the optimal $\sigma$, and the fish's positions can be extracted (**pangeo-fish** features several options for reconstructing trajectories):

```python
from pangeo_fish.helpers import optimize_pdf, predict_positions

parameters = optimize_pdf(combined_pdf, maximum_speed, save_parameters=True, ...)
states, trajectories = predict_positions(path_to_previous_results, track_modes, ...)
```

## Result analysis and visualization

Every result of the step described above can be easily visualized for analysis with **pangeo-fish**.
For instance, the time series of the archival tag's can be easily visualized with:

```python
from pangeo_fish.helpers import plot_tag

plot = plot_tag(tag, tag_log, save_html=True, ...)
```

The user can also plot an interactive visualization of the trajectories:

```python
from pangeo_fish.helpers import plot_trajectories

plot = plot_trajectories(path_to_previous_results, track_modes, save_html=True)
```

As for the state and emission distributions, a video of their evolution (alongside of each other) can be rendered:

```python
from pangeo_fish.helpers import open_distributions, render_distributions

data = open_distributions(path_to_previous_results, storage_options, ...)
render_distributions(data, "results/", extension="mp4", remove_frames=True, ...)
```

# Conclusion

**pangeo-fish** is a Python package that implements a geolocation model, based on a Hidden Markov Model, for estimating fish positions from archival tag and oceanic data.
Designed to work with the Pangeo ecosystem, it aims to support the ecologists with their research, by handling backend processes — such as data loading or parallel computation — while exposing a user-friendly interface to manage their archival tag data and run the geolocation model.

# Acknowledgements

- T Odaka, JM Delouis and J Magin are supported by the CNES Appel, a projet R&T R-S23/DU-0002-025-01.
- T Odaka, JM Delouis and M Woillez are supported by the TAOS project funded by the IFREMER via the AMII OCEAN 2100 programme.
- Q Mazouni, M Woillez, A Fouilloux and T Odaka are supported by Global Fish Tracking System (GFTS), a Destination Earth use case procured and funded by ESA (SA Contract No. 4000140320/23/I-NS, DESTINATION EARTH USE CASES - DESP USE CASES - ROUND (YEAR 2023))
- M Woillez and T Odaka are supported by Digital Twin of the Ocean - Animal Tracking (DTO Track), a European project that aims to create a digital twin of the North Sea using animal tracking data and funded by the Sustainable Blue Economy Partnership (ANR-24-SBEP-0001)

# References
