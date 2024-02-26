import json
import pathlib

import pint_xarray
import rich_click as click

from pangeo_fish.io import open_tag

from .cluster import create_cluster

ureg = pint_xarray.unit_registry

click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True


def decode_parameters(obj):
    if list(obj) != ["magnitude", "units"]:
        return obj

    return ureg.Quantity(obj["magnitude"], obj["units"])


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
    client = create_cluster(dask_cluster)

    decoded = json.load(parameters, object_hook=decode_parameters)

    tag_root = decoded["paths"]["tag_root"]
    catalog_url = decoded["paths"]["catalog_url"]

    parameters = decoded["parameters"]

    tag = open_tag(tag_root, parameters["tag_name"])


@main.command("estimate", short_help="estimate the model parameter")
def estimate():
    pass


@main.command("decode", short_help="produce the model output")
def decode():
    pass
