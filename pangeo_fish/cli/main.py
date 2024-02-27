import json
import pathlib

import pint_xarray
import rich_click as click

from pangeo_fish.cli.cluster import create_cluster
from pangeo_fish.cli.path import construct_target_root

ureg = pint_xarray.unit_registry

click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True


def decode_parameters(obj):
    if list(obj) != ["magnitude", "units"]:
        return obj

    return ureg.Quantity(obj["magnitude"], obj["units"])


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
@click.option("--cluster-definition", type=click.File(mode="r"))
@click.option(
    "--compute/--no-compute",
    type=bool,
    default=True,
    help="load the emission pdf into memory before the parameter estimation",
)
@click.argument("parameters", type=click.File(mode="r"))
@click.argument("runtime_config", type=click.File(mode="r"))
def estimate(parameters, runtime_config, cluster_definition, compute):
    import xarray as xr

    from pangeo_fish.hmm.estimator import EagerScoreEstimator
    from pangeo_fish.hmm.optimize import EagerBoundsSearch
    from pangeo_fish.pdf import combine_emission_pdf

    runtime_config = json.load(runtime_config)
    parameters = json.load(parameters, object_hook=decode_parameters)
    cluster_definition = json.load(cluster_definition)

    target_root = construct_target_root(runtime_config, parameters)

    with create_cluster(**cluster_definition) as client:
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
