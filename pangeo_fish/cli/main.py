import json
from contextlib import nullcontext

import pint_xarray
import rich_click as click
import xarray as xr
from rich.console import Console

from pangeo_fish.cli.cluster import create_cluster
from pangeo_fish.cli.path import construct_target_root
from pangeo_fish.pdf import combine_emission_pdf

ureg = pint_xarray.unit_registry

click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
console = Console()


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
@click.option("--cluster-definition", type=click.File(mode="r"))
@click.argument("parameters", type=click.File(mode="r"))
@click.argument("runtime_config", type=click.File(mode="r"))
def prepare(parameters, runtime_config, cluster_definition):
    """transform the input data into a set of emission parameters"""
    import intake

    from pangeo_fish import acoustic
    from pangeo_fish.cli.prepare import (
        regrid,
        subtract_data,
        temperature_emission_matrices,
    )
    from pangeo_fish.io import open_copernicus_catalog, open_tag

    runtime_config = json.load(runtime_config)
    parameters = json.load(parameters, object_hook=decode_parameters)
    cluster_definition = json.load(cluster_definition)

    target_root = construct_target_root(runtime_config, parameters)
    target_root.mkdir(parents=True, exist_ok=True)

    with create_cluster(**cluster_definition) as client, console.status(
        "[bold blue]processing[/]"
    ) as status:
        console.print(f"[bold white]dashboard link[/]: {client.dashboard_link}")

        # open tag
        tag = open_tag(runtime_config["tag_root"], parameters["tag_name"])
        console.log("successfully opened tag log")

        # open model
        cat = intake.open_catalog(runtime_config["catalog_url"])
        model = open_copernicus_catalog(cat)
        console.log("successfully opened reference model")

        status.update(
            "[bold blue]compare temperature from reference model and tag log[/]"
        )
        differences = subtract_data(tag, model, parameters)
        differences.chunk({"time": 1, "lat": -1, "lon": -1}).to_zarr(
            f"{target_root}/diff.zarr", mode="w", consolidated=True
        )
        console.log("stored temperature differences")

        # open back the diff
        differences = (
            xr.open_dataset(f"{target_root}/diff.zarr", engine="zarr", chunks={})
            .pipe(lambda ds: ds.merge(ds[["latitude", "longitude"]].compute()))
            .swap_dims({"lat": "yi", "lon": "xi"})
            .drop_vars(["lat", "lon"])
        )
        console.log("reopened temperature differences")

        status.update("[bold blue]verifying result[/]")
        counts = differences["diff"].count(["xi", "yi"]).compute()
        console.log("finished detecting missing time slices")
        if (counts == 0).any():
            raise click.ClickException(
                "some time slices are 0. Try rerunning the step or"
                " checking the connection to the data server."
            )
        console.log("detecting missing time slices: none found")

        status.update("[bold blue]regridding[/]")
        regridded = regrid(differences, parameters)
        regridded.chunk({"x": -1, "y": -1, "time": 1}).to_zarr(
            f"{target_root}/diff-regridded.zarr",
            mode="w",
            consolidated=True,
        )
        console.log("finished regridding")

        # temperature emission matrices
        differences = xr.open_dataset(
            f"{target_root}/diff-regridded.zarr", engine="zarr", chunks={}
        )

        status.update("[bold blue]constructing emission matrices from temperature[/]")
        emission = temperature_emission_matrices(differences, tag, parameters)
        emission.chunk({"x": -1, "y": -1, "time": 1}).to_zarr(
            f"{target_root}/emission.zarr",
            mode="w",
            consolidated=True,
        )
        console.log("finished constructing emission matrices from temperature data")

        del differences

        # acoustic emission matrices
        emission = xr.open_dataset(
            f"{target_root}/emission.zarr", engine="zarr", chunks={}
        )
        status.update(
            "[bold blue]constructing emission matrices from acoustic detections[/]"
        )
        combined = emission.merge(
            acoustic.emission_probability(
                tag,
                emission[["time", "cell_ids", "mask"]].compute(),
                parameters["receiver_buffer"],
            )
        )

        combined.chunk({"x": -1, "y": -1, "time": 1}).to_zarr(
            f"{target_root}/emission-acoustic.zarr", mode="w", consolidated=True
        )
        console.log("finished writing emission matrices from acoustic detections")

        del combined


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
    from pangeo_fish.hmm.estimator import EagerScoreEstimator
    from pangeo_fish.hmm.optimize import EagerBoundsSearch

    runtime_config = json.load(runtime_config)
    parameters = json.load(parameters, object_hook=decode_parameters)
    cluster_definition = json.load(cluster_definition)

    target_root = construct_target_root(runtime_config, parameters)

    if compute:
        chunks = None
        client = nullcontext()
    else:
        chunks = {"x": -1, "y": -1}
        client = create_cluster(**cluster_definition)
        console.print(f"dashboard link: {client.dashboard_link}")

    with client, console.status(
        "[bold blue]estimating the model parameter...[/]"
    ) as status:
        emission = (
            xr.open_dataset(
                f"{target_root}/emission-acoustic.zarr",
                engine="zarr",
                chunks=chunks,
                inline_array=True,
            )
            .pipe(combine_emission_pdf)
            .pipe(maybe_compute, compute=compute)
        )
        console.log("opened emission probabilities")

        console.status("[bold blue]detecting missing timesteps[/]")
        counts = emission["pdf"].count(["y", "x"]).compute()
        if (counts == 0).any():
            raise click.ClickException(
                "Some time slices are all-nan, which will cause the optimization to fail."
                " This can happen if a component of the emission probability matrices has"
                " all-nan time slices, or if the components don't have overlaps in"
                " all-nan areas."
            )
        console.log("detecting missing timesteps: none found")

        # TODO: make estimator and optimizer configurable somehow
        estimator = EagerScoreEstimator()
        optimizer = EagerBoundsSearch(
            estimator,
            (1e-4, emission.attrs["max_sigma"]),
            optimizer_kwargs={"disp": 3, "xtol": parameters.get("tolerance", 0.01)},
        )

        status.update("[bold blue]searching for optimal model parameters[/]")
        optimized = optimizer.fit(emission)
        console.log("model parameter: completed search")

        if optimized.sigma == emission.attrs["max_sigma"]:
            raise click.ClickException(
                "Found the upper limit of the parameter search space."
                " Make sure the search space is big enough."
            )
        console.log("model parameter: checks passed")

        status.update("[bold blue]storing the optimized model parameter[/]")
        params = optimized.to_dict()
        with target_root.joinpath("parameters.json").open(mode="w") as f:
            json.dump(params, f)
        console.log("model parameter: finished writing the optimized model parameter")


@main.command("decode", short_help="produce the model output")
def decode():
    pass
