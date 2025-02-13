import inspect
import fsspec
import numpy as np
import pint
import s3fs
import intake
import xdggs

from movingpandas import TrajectoryCollection
from typing import Dict, List, Tuple, Union

import holoviews as hv
# import hvplot.xarray
import xarray as xr
import pandas as pd

from pangeo_fish.cf import bounds_to_bins
from pangeo_fish.acoustic import emission_probability
from pangeo_fish.diff import diff_z
from pangeo_fish.grid import center_longitude
from xarray_healpy import HealpyGridInfo, HealpyRegridder
from pangeo_fish.io import open_copernicus_catalog, open_tag, prepare_dataset, read_trajectories, save_html_hvplot, save_trajectories
from pangeo_fish.tags import adapt_model_time, reshape_by_bins, to_time_slice
from pangeo_fish.hmm.prediction import Gaussian1DHealpix, Gaussian2DCartesian
from pangeo_fish.hmm.estimator import EagerEstimator
from pangeo_fish.hmm.optimize import EagerBoundsSearch
from pangeo_fish.utils import temporal_resolution
from pangeo_fish.visualization import plot_map
from pangeo_fish.pdf import combine_emission_pdf, normal

import pangeo_fish.distributions as distrib

from toolz.functoolz import curry # to change
from toolz.dicttoolz import valfilter


__all__ = [
    "load_tag",
    "plot_tag",
    "load_model",
    "compute_diff",
    "open_diff_dataset",
    "regrid_dataset",
    "compute_emission_pdf",
    "compute_acoustic_pdf",
    "combine_pdfs",
    "optimize_pdf",
    "predict_positions",
    "plot_trajectories",
    "open_distributions",
    "plot_distributions"
]


def _inspect_curry_obj(curried_obj):
    """Inspect a curried object and retrieves its args and kwargs."""
    sig = inspect.signature(curried_obj.func)
    # default parameters
    params = {
        k: v.default if v.default is not inspect.Parameter.empty else None
            for k, v in sig.parameters.items()

    }

    # sig.parameters is ordered, so we can add the args of `curried_obj`
    for arg_name, arg_value in zip(sig.parameters.keys(), curried_obj.args):
        params[arg_name] = arg_value

    # finally, updates with the keywords (if any)
    params.update(curried_obj.keywords)
    return params


def _update_params_dict(factory, params: Dict):
    """Inspect `factory` (assumed to be a curried object) to get its kw/args and update `params`.

    Note that `params` is updated with string representations of the arguments retrieved **except `cell_ids`**.

    Parameters
    -----------
    factory : Curried object
        It must have the attributes `func`, `args`, and `keywords`
    params : Dict
        The dictionary to update

    Returns
    --------
    params : Dict
        The updated dictionary
    """

    kwargs = {k: str(v) for k, v in _inspect_curry_obj(factory).items() if k != "cell_ids"}

    params["predictor_factory"] = {
        "class": str(factory),
        "kwargs": kwargs
    }

    return params


def to_healpix(ds: xr.Dataset) -> xr.Dataset:
    """Helper that loads a Dataset as a HEALPix grid (indexed by "cell_ids")."""
    ds["cell_ids"].attrs["grid_name"] = "healpix"
    attrs_to_keep = ["level", "grid_name"]
    ds["cell_ids"].attrs = {
        key: value
        for (key, value) in ds["cell_ids"].attrs.items()
        if key in attrs_to_keep
    }
    return ds.pipe(xdggs.decode)


def regrid_to_2d(ds: xr.Dataset):
    grid = HealpyGridInfo(level=ds.dggs.grid_info.level)
    return grid.to_2d(ds)


def load_tag(tag_root, tag_name):
    """load a tag.

    Returns
    -------
    tag : xr.DataTree
        The tag
    tag_log : xr.Dataset
        The DST data (sliced w.r.t released and recapture dates)
    time_slice : slice or like
        Time interval described by the released and recapture dates
    """
    tag = open_tag(tag_root, tag_name)
    time_slice = to_time_slice(tag["tagging_events/time"])
    tag_log = tag["dst"].ds.sel(time=time_slice).assign_attrs({"tag_name": tag_name})
    return tag, tag_log, time_slice


def plot_tag(tag: xr.DataTree, tag_log: xr.Dataset):
    """plot a tag.

    Parameters
    ----------
    tag : xr.DataTree
        The tag
    tag_log : xr.Dataset
        The DST data

    # target_root : str, optional
    #     Root of the folder to save `plot` as a HMTL file `tags.html`.

    Returns
    -------
    hvplot : hv.Overlay
        Interactive plot
    """
    plot = (
        (-tag["dst"].pressure).hvplot(width=1000, height=500, color="blue")
        * (-tag_log).hvplot.scatter(
            x="time", y="pressure", color="red", size=5, width=1000, height=500
        )
        * (
            (tag["dst"].temperature).hvplot(width=1000, height=500, color="blue")
            * (tag_log).hvplot.scatter(
                x="time", y="temperature", color="red", size=5, width=1000, height=500
            )
        )
    )
    return plot


def _open_copernicus_model(catalog_url="https://data-taos.ifremer.fr/kerchunk/ref-copernicus.yaml"):
    cat = intake.open_catalog(catalog_url)
    model = open_copernicus_catalog(cat)
    return model


def _open_reference_model():
    parquet_path_url = "s3://gfts-reference-data/NORTHWESTSHELF_ANALYSIS_FORECAST_PHY_004_013/combined_2022_to_2024.parq/"
    target_opts = {"anon": False}
    remote_opts = {"anon": False}
    fs = fsspec.filesystem(
        "reference",
        fo=parquet_path_url,
        remote_protocol="s3",
        remote_options=remote_opts,
        target_options=target_opts,
    )
    m = fs.get_mapper("")

    reference_ds = xr.open_dataset(
        m, engine="zarr", chunks={}, backend_kwargs={"consolidated": False}
    )
    reference_ds.coords["depth"].values[0] = 0.
    return reference_ds


def load_model(tag_log: xr.Dataset, time_slice: slice, bbox: Dict[str, Tuple[float, float]], chunk_time=24):
    """load and prepare the reference model for a computation."""
    # reference_ds = _open_reference_model()
    # model = prepare_dataset(reference_ds)
    model = _open_copernicus_model()
    reference_model = (
        model.sel(time=adapt_model_time(time_slice))
        .sel(lat=slice(*bbox["latitude"]), lon=slice(*bbox["longitude"]))
        .pipe(
            lambda ds: ds.sel(
                depth=slice(None, (tag_log["pressure"].max() - ds["XE"].min()).compute())
            )
        )
    ).chunk({"time": chunk_time, "lat": -1, "lon": -1, "depth": -1})
    return reference_model



def compute_diff(reference_model: xr.Dataset, tag_log: xr.Dataset, relative_depth_threshold: float, chunk_time=24):
    """compute the difference between the reference model and the DST data of a tag."""
    reshaped_tag = reshape_by_bins(
        tag_log,
        dim="time",
        bins=(
            reference_model.cf.add_bounds(["time"], output_dim="bounds")
            .pipe(bounds_to_bins, bounds_dim="bounds")
            .get("time_bins")
        ),
        bin_dim="bincount",
        other_dim="obs",
    ).chunk({"time": chunk_time})
    attrs = tag_log.attrs.copy()
    attrs.update(
        {
            "relative_depth_threshold": relative_depth_threshold
        }
    )
    diff = (
        diff_z(reference_model, reshaped_tag, depth_threshold=relative_depth_threshold)
        .assign_attrs(attrs)
        .assign(
            {
                "H0": reference_model["H0"],
                "ocean_mask": reference_model["H0"].notnull(),
            }
        )
    )
    return diff


def open_diff_dataset(target_root: str, storage_options: dict):
    """open a diff dataset.

    Parameters
    ----------
    target_root : str
        Path root where to find `diff.zarr`
    storage_options : dict
        Additional information for `xarray` to open the `.zarr` array

    Returns
    -------
    ds : xr.Dataset
        The dataset
    """
    ds = (
        xr.open_dataset(
            f"{target_root}/diff.zarr",
            engine="zarr",
            chunks={},
            storage_options=storage_options,
        )
        .pipe(lambda ds: ds.merge(ds[["latitude", "longitude"]].compute()))
        .swap_dims({"lat": "yi", "lon": "xi"})
        .drop_vars(["lat", "lon"])
    )
    return ds


def regrid_dataset(ds: xr.Dataset, nside, min_vertices=1, rot={"lat": 0, "lon": 0}, dims: List[str] = ["x", "y"]):
    """regrids a dataset as a HEALPix grid, whose primary advantage is that all its cells/pixels cover the same surface area.
    It currently only supports 2d regridding, i.e., ["x", "y"] indexing.

    Parameters
    ----------
    tag_log : xr.Dataset
        The DST data
    nside : int
        Tesolution of the HEALPix grid
    min_vertices : int, default to 1
        Minimum number of vertices for a valid transcription
    rot : Dict[str, Tuple[float, float]], default to {"lat": 0, "lon": 0}
        Mapping of angles to rotate the HEALPix grid. It must contain the keys "lon" and "lat"
    dims : List[str], default to ["x", "y"]
        The list of the dimensions for the regridding. Either ["x", "y"] or ["cells"]

    Returns
    -------
    reshaped : xr.Dataset
        HEALPix version of `ds`
    """
    grid = HealpyGridInfo(level=int(np.log2(nside)), rot=rot)
    target_grid = grid.target_grid(ds).pipe(center_longitude, 0)
    regridder = HealpyRegridder(
        ds[["longitude", "latitude", "ocean_mask"]],
        target_grid,
        method="bilinear",
        interpolation_kwargs={"mask": "ocean_mask", "min_vertices": min_vertices},
    )
    regridded = regridder.regrid_ds(ds)
    if dims == ["x", "y"]:
        reshaped = grid.to_2d(regridded).pipe(center_longitude, 0)
    elif dims == ["cells"]:
        reshaped = regridded.assign_coords(
            cell_ids=lambda ds: ds.cell_ids.astype("int64")
        )
    else:
        raise ValueError(f"Unknown dims \"{dims}\".")

    # adds the attributes found in `ds` as well as `min_vertices`
    attrs = ds.attrs.copy()
    attrs.update({"min_vertices": min_vertices})
    reshaped = reshaped.assign_attrs(attrs)
    return reshaped


def compute_emission_pdf(diff_ds: xr.Dataset, events_ds: xr.Dataset, differences_std: float, recapture_std: float, dims: List[str] = ["x", "y"]):
    """compute the temporal emission matrices given a dataset and tagging events.

    Parameters
    ----------
    diff_ds : xr.Dataset
        A dataset that must have the variables `diff` and `ocean_mask`
    events_ds : xr.Dataset
        The tagging events. It must have the coordinate `event_name` and values `release` and `fish_death`
    differences_std : float
        Standard deviation that is applied to the data (passed to `scipy.stats.norm.pdf`). It'd express the estimated certainty of the field of difference
    recapture_std : float
        Covariance for the recapture event. It should reflect the certainty of the final recapture area
    dims : List[str], default to ["x", "y"]
        The list of the dimensions. Either ["x", "y"] or ["cells"]

    Returns
    -------
    emission_pdf : xr.Dataset
        The emission pdf
    """
    if dims == ["x", "y"]:
        is_2d = True
    elif dims == ["cells"]:
        is_2d = False
    else:
        raise ValueError(f"Unknown dims \"{dims}\".")


    if not is_2d:
        diff_ds = to_healpix(diff_ds)


    grid = diff_ds[["latitude", "longitude"]].compute()
    initial_position = events_ds.sel(event_name="release")
    final_position = events_ds.sel(event_name="fish_death")

    if dims == ["x", "y"]:
        cov = distrib.create_covariances(1e-6, coord_names=["latitude", "longitude"])
        initial_probability = distrib.normal_at(
            grid, pos=initial_position, cov=cov, normalize=True, axes=["latitude", "longitude"]
        )
    else:
        initial_probability = distrib.healpix.normal_at(grid, pos=initial_position, sigma=1e-3)


    if final_position[["longitude", "latitude"]].to_dataarray().isnull().all():
        final_probability = None
    else:
        if is_2d:
            cov = distrib.create_covariances(recapture_std**2, coord_names=["latitude", "longitude"])
            final_probability = distrib.normal_at(
                grid,
                pos=final_position,
                cov=cov,
                normalize=True,
                axes=["latitude", "longitude"],
            )
        else:
            final_probability = distrib.healpix.normal_at(grid, pos=final_position, sigma=recapture_std)


    emission_pdf = (
        normal(diff_ds["diff"], mean=0, std=differences_std, dims=dims)
        .to_dataset(name="pdf")
        .assign(
            valfilter(
                lambda x: x is not None,
                {
                    "initial": initial_probability,
                    "final": final_probability,
                    "mask": diff_ds["ocean_mask"],
                },
            )
        )
    )  # type: xr.Dataset
    attrs = diff_ds.attrs.copy()
    attrs.update(
        {
            "differences_std": differences_std,
            "recapture_std": recapture_std
        }
    )
    emission_pdf = emission_pdf.assign_attrs(attrs)
    return emission_pdf # chunk?


def compute_acoustic_pdf(emission_ds: xr.Dataset, tag: xr.DataTree, receiver_buffer: pint.Quantity, chunk_time=24, dims: List[str] = ["x", "y"]):
    """compute emission probability maps from (acoustic) detection data.

    Parameters
    ----------
    emission_ds : xr.Dataset
        A dataset that must have the variables `time`, `mask` and `cell_ids`
    tag : xr.DataTree
        The tag data. It must have the datasets `acoustic` and `stations`
    receiver_buffer : pint.Quantity
        Maximum allowed detection distance for acoustic receivers
    chunk_time : int, default to 24
        Chunk size for the time dimension
    dims : List[str], default to ["x", "y"]
        The list of the dimensions. Either ["x", "y"] or ["cells"]

    Returns
    -------
    acoustic_pdf : xr.Dataset
        The emission pdf
    """
    if dims == ["cells"]:
        lon, lat = emission_ds["cell_ids"].attrs.get("lat", 0), emission_ds["cell_ids"].attrs.get("lon", 0)
        emission_ds = to_healpix(emission_ds)
        # adds back "lon" and "lat" keys
        emission_ds["cell_ids"].attrs["lon"] = lon
        emission_ds["cell_ids"].attrs["lat"] = lat

    acoustic_pdf = emission_probability(
        tag,
        emission_ds[["time", "cell_ids", "mask"]].compute(),
        receiver_buffer,
        nondetections="mask",
        cell_ids="keep",
        chunk_time=chunk_time,
        dims=dims
    )
    attrs = emission_ds.attrs.copy()
    attrs.update(
        {
            "receiver_buffer": str(receiver_buffer)
        }
    )
    acoustic_pdf = acoustic_pdf.assign_attrs(attrs)
    return acoustic_pdf


def combine_pdfs(emission_ds: xr.Dataset, acoustic_ds: xr.Dataset, chunks):
    """combine and normalize 2 pdfs.

    Parameters
    ----------
    emission_ds : xr.Dataset
        Dataset of emission probabilities
    acoustic_ds : xr.Dataset
        Dataset of acoustic probabilities
    chunks : dict
        How to chunk the data

    Returns
    -------
    combined : xr.Dataset
        The combined pdf
    """
    combined = emission_ds.merge(acoustic_ds)
    combined = (
        combined.pipe(combine_emission_pdf)
        .chunk(chunks)
    )
    return combined


def _get_predictor_factory(ds: xr.Dataset, truncate: float, dims: List[str]):
    if dims == ["x", "y"]:
        predictor = curry(Gaussian2DCartesian, truncate=truncate)
    elif dims == ["cells"]:
        predictor = curry(
            Gaussian1DHealpix,
            cell_ids=ds["cell_ids"].data,
            grid_info=ds.dggs.grid_info,
            truncate=truncate,
            weights_threshold=1e-8,
            pad_kwargs={"mode": "constant", "constant_value": 0},
            optimize_convolution=True,
        )
    else:
        raise ValueError(f"Unknown dims \"{dims}\".")
    return predictor


def _get_max_sigma(ds: xr.Dataset, earth_radius: pint.Quantity, adjustment_factor: float, truncate: float, maximum_speed: pint.Quantity, as_radians: bool) -> float:
    earth_radius_ = xr.DataArray(earth_radius, dims=None)

    timedelta = temporal_resolution(ds["time"]).pint.quantify().pint.to("h")
    grid_resolution = earth_radius_ * ds["resolution"].pint.quantify()

    maximum_speed_ = xr.DataArray(maximum_speed, dims=None).pint.to("km / h")
    if as_radians:
        max_grid_displacement = maximum_speed_ * timedelta * adjustment_factor / earth_radius_
    else: # in pixels
        max_grid_displacement = maximum_speed_ * timedelta * adjustment_factor / grid_resolution
    max_sigma = max_grid_displacement.pint.to("dimensionless").pint.magnitude / truncate

    return max_sigma.item()


def optimize_pdf(
        ds: xr.Dataset,
        earth_radius: pint.Quantity,
        adjustment_factor: float,
        truncate: float,
        maximum_speed: pint.Quantity,
        tolerance: float,
        dims: List[str] = ["x", "y"]
    ) -> dict:
    """optimize a temporal emission pdf.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset of emission probabilities
    earth_radius : pint.Quantity
        Radius of the Earth used for distance calculations
    adjustment_factor : float
        Factor value for the maximum fish's displacement
    truncate : float
        Truncating factor for convolution process
    maximum_speed : pint.Quantity
        Maximum fish's velocity
    tolerance : float
        Tolerance level for the optimised parameter search computation
    dims : List[str], default to ["x", "y"]
        The list of the dimensions. Either ["x", "y"] or ["cells"]

    Returns
    -------
    params : dict
        A dictionary containing the optimization results (mainly, the sigma value of the Brownian movement model)
    """
    # it is important to compute before re-indexing?
    ds = ds.compute()

    if "cells" in ds.dims:
        ds = to_healpix(ds)
        as_radians = True
    else:
        as_radians = False

    max_sigma = _get_max_sigma(ds, earth_radius, adjustment_factor, truncate, maximum_speed, as_radians)
    predictor_factory = _get_predictor_factory(ds=ds, truncate=truncate, dims=dims)

    estimator = EagerEstimator(sigma=None, predictor_factory=predictor_factory)
    ds.attrs["max_sigma"] = max_sigma # limitation of the helper

    optimizer = EagerBoundsSearch(
        estimator,
        (1e-4, ds.attrs["max_sigma"]),
        optimizer_kwargs={"disp": 3, "xtol": tolerance},
    )
    optimized = optimizer.fit(ds)
    params = optimized.to_dict()  # type: dict
    _update_params_dict(factory=predictor_factory, params=params)
    return params


def predict_positions(
        target_root: str,
        storage_options: dict,
        chunks: dict,
        track_modes=["mean", "mode"],
        additional_track_quantities=["speed", "distance"],
        dims: List[str] = ["x", "y"]
        ):
    """high-level helper function for predicting fish's positions and generating the consequent trajectories.
    It futhermore saves the latter under `states.zarr` and `trajectories.parq`.

    Parameters
    ----------
    target_root : str
        Path to a folder that must contain the `combined.zarr` and `parameters.json` files.
        .. warning::
            it must not end with "/".
    storage_options : dict
        Additional information for `xarray` to open the `.zarr` array
    chunks : dict
        Chunk size upon loading the pdf xr.Dataset
    track_modes : list[str], default to ["mean", "mode"]
        Options for decoding trajectories. See `pangeo_fish.hmm.estimator.EagerEstimator.decode`
    additional_track_quantities : list[str], default to ["speed", "distance"]
        Additional quantities to compute from the decoded tracks. See `pangeo_fish.hmm.estimator.EagerEstimator.decode`
    dims : List[str], default to ["x", "y"]
        The list of the dimensions. Either ["x", "y"] or ["cells"]

    Returns
    -------
    states : xr.Dataset
        A geolocation model, i.e., positional temporal probabilities
    trajectories : TrajectoryCollection
        The tracks decoded from `states`
    """
    # loads the normalized .zarr array
    emission = xr.open_dataset(
        f"{target_root}/combined.zarr",
        engine="zarr",
        chunks=chunks,
        inline_array=True,
        storage_options=storage_options,
    )
    emission = emission.compute()

    if "cells" in emission.dims:
        emission = to_healpix(emission)
    else:
        assert all([d in emission.dims for d in dims]), f"Not all the dimensions provided (dims=\"{dims}\") were found in the emission distribution."
        emission["pdf"] = emission["pdf"].transpose("time", "y", "x")


    params = pd.read_json(
        f"{target_root}/parameters.json", storage_options=storage_options
    ).to_dict()[0]

    # do not account for the other kwargs...
    # not very robust yet...
    truncate = float(params["predictor_factory"]["kwargs"]["truncate"])
    predictor_factory = _get_predictor_factory(emission, truncate=truncate, dims=dims)

    optimized = EagerEstimator(sigma=params["sigma"],  predictor_factory=predictor_factory)

    states = optimized.predict_proba(emission)
    states = states.to_dataset().chunk(chunks)  # type: xr.Dataset
    states.attrs.update(emission.attrs)

    states.to_zarr(
        f"{target_root}/states.zarr",
        mode="w",
        consolidated=True,
        storage_options=storage_options,
    )

    trajectories = optimized.decode(
        emission,
        states.fillna(0),
        mode=track_modes,
        progress=False,
        additional_quantities=additional_track_quantities,
    )

    save_trajectories(trajectories, target_root, storage_options, format="parquet")

    return states, trajectories


def plot_trajectories(target_root: str, track_modes: list[str], storage_options: dict, save_html=True):
    """read decoded trajectories and plots an interactive visualization.
    Optionally, the plot can be saved as a HTML file.

    Parameters
    ----------
    target_root : str
        Path to a folder that must contain a trajectory collection `trajectories.parq`
        .. warning::
            it must not end with "/".
    track_modes : list[str]
        Names of the tracks
    storage_options : dict
        Additional information for `xarray` to open the `.zarr` array
    save_html : bool, default to True
        Whether to save the plot (under `{target_root}/trajectories.html`)

    Returns
    -------
    plot : hv.Layout
        Interactive plot of the trajectories
    """
    trajectories = read_trajectories(
        track_modes, target_root, storage_options, format="parquet"
    )
    plots = [
        traj.hvplot(
            c="speed",
            tiles="CartoLight",
            title=traj.id,
            cmap="cmo.speed",
            width=300,
            height=300,
        )
        for traj in trajectories.trajectories
    ]
    plot = hv.Layout(plots).cols(2)

    if save_html:
        filepath = f"{target_root}/trajectories.html"
        save_html_hvplot(plot, filepath, storage_options)

    return plot


def open_distributions(target_root: str, storage_options: dict, chunk_time=24):
    """load and merge the `emission` and `states` distributions into a single dataset.

    Parameters
    ----------
    target_root : str
        Path to a folder that must contain the `combined.zarr` and `states.zarr` files.
        .. warning::
            it must not end with "/".
    storage_options : dict
        Additional information for `xarray` to open the `.zarr` array
    chunk_time : int, default to 24
        Chunk size for the time dimension

    Returns
    -------
    data : xr.Dataset
        The merged and cleaned dataset
    """
    emission = (
        xr.open_dataset(
            f"{target_root}/combined.zarr",
            engine="zarr",
            chunks={},
            inline_array=True,
            storage_options=storage_options,
        )
        .rename_vars({"pdf": "emission"})
        .drop_vars(["final", "initial"])
    )
    states = xr.open_dataset(
        f"{target_root}/states.zarr",
        engine="zarr",
        chunks={},
        inline_array=True,
        storage_options=storage_options,
    ).where(emission["mask"])

    data = xr.merge([states, emission.drop_vars(["mask"])])

    # if the data is 1D indexed, regrids it to 2D
    # since this function is expected to be used for plotting tasks
    if "cells" in data.dims:
        data = to_healpix(data)
        data = regrid_to_2d(data)

    data = data.assign_coords(longitude=((data.longitude + 180) % 360 - 180))
    data = data.chunk({"time": chunk_time})

    return data


def plot_distributions(data: xr.Dataset, bbox=None):
    """plot an interactive visualization of both a dataset resulting from the merging of `emission` and the `states` distributions.
    See `pangeo_fish.helpers.open_distributions()`.

    Parameters
    ----------
    data : xr.Dataset
        A dataset that contains the `emission` and `states` variables
    bbox : dict[str, tuple[float, float]], optional
        The spatial boundaries of the area of interest. Shoud have the keys "longitude" and "latitude".

    Returns
    -------
    plot : hv.Layout
        Interactive plot of the `states` and `emission` distributions
    """
    plot1 = plot_map(data["states"], bbox)
    plot2 = plot_map(data["emission"], bbox)
    plot = hv.Layout([plot1, plot2]).cols(2)

    return plot