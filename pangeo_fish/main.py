import json
import pathlib

import pint_xarray
import rich_click as click

from pangeo_fish.cluster import create_cluster

ureg = pint_xarray.unit_registry

click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True


def decode_parameters(obj):
    if list(obj) != ["magnitude", "units"]:
        return obj

    return ureg.Quantity(obj["magnitude"], obj["units"])


def construct_target_root(runtime_config, parameters):
    def default_scheme(params):
        return params["tag_name"]

    def subdir(params):
        rot = "_".join(str(params["rot"][k]) for k in ["lon", "lat"])
        nside = params["nside"]
        thresh = params["relative_depth_threshold"]
        buffer_size = params["receiver_buffer"].m_as("m")
        return f"{nside}-{thresh}-{rot}-{buffer_size}/{params['tag_name']}"

    naming_schemes = {
        "default": default_scheme,
        "subdir": subdir,
    }

    naming_scheme = runtime_config.get("naming_scheme", "default")
    if naming_scheme not in naming_schemes:
        raise ValueError(f"unknown naming scheme: {naming_scheme}")

    formatter = naming_schemes[naming_scheme]

    scratch_root = pathlib.Path(runtime_config["scratch_root"])
    target_root = scratch_root.joinpath(formatter(parameters))

    return target_root


def maybe_compute(ds, compute):
    if not compute:
        return ds

    return ds.compute()


@click.group()
def main():
    """Run the pangeo-fish model."""
    pass


@main.command(
    "prepare",
    short_help="transform the data into something suitable for the model to run",
)
@click.option(
    "--dask-cluster",
    type=str,
    default="local",
    help="dask cluster to run on. May be 'local', a scheduler address, or the path to a file containing 'dask_jobqueue' parameters",
)
@click.argument("parameters", type=click.File(mode="r"))
@click.argument("scratch_root", type=click.Path(path_type=pathlib.Path, writable=True))
def prepare(parameters, scratch_root, dask_cluster):
    """transform the input data into a set of emission parameters"""
    pass


@main.command("estimate", short_help="estimate the model parameter")
@click.option(
    "--compute/--no-compute",
    type=bool,
    default=True,
    help="load the emission pdf into memory before the parameter estimation",
)
@click.argument("parameters", type=click.File(mode="r"))
@click.argument("runtime_config", type=click.File(mode="r"))
def estimate(parameters, runtime_config, compute):
    import xarray as xr

    from pangeo_fish.hmm.estimator import EagerScoreEstimator
    from pangeo_fish.hmm.optimize import EagerBoundsSearch
    from pangeo_fish.pdf import combine_emission_pdf

    runtime_config = json.load(runtime_config)
    parameters = json.load(parameters, object_hook=decode_parameters)
    target_root = construct_target_root(runtime_config, parameters)

    with create_cluster(**runtime_config["dask-cluster"]) as client:
        print(f"dashboard link: {client.dashboard_link}", flush=True)

        emission = (
            xr.open_dataset(
                f"{target_root}/emission.zarr",
                engine="zarr",
                chunks={"x": -1, "y": -1, "time": 1},
                inline_array=True,
            )
            .pipe(combine_emission_pdf)
            .pipe(maybe_compute, compute=compute)
        )

        # TODO: make this estimator and optimizer configurable somehow
        estimator = EagerScoreEstimator()
        optimizer = EagerBoundsSearch(
            estimator,
            (1e-4, emission.attrs["max_sigma"]),
            optimizer_kwargs={"disp": 3, "xtol": parameters.get("tolerance", 0.01)},
        )

        optimized = optimizer.fit(emission)

    params = optimized.to_dict()
    with target_root.joinpath("parameters.json").open(mode="w") as f:
        json.dump(params, f)


@main.command("decode", short_help="produce the model output")
def decode():
    pass
