import json
import pathlib

import pint_xarray
import rich_click as click

from pangeo_fish.io import open_tag

ureg = pint_xarray.unit_registry

click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True


def decode_parameters(f):
    def decoder(obj):
        if list(obj) != ["magnitude", "units"]:
            return obj

        return ureg.Quantity(obj["magnitude"], obj["units"])

    return json.load(f, object_hook=decoder)


def connect_to_cluster(cluster_config):
    if cluster_config == "local":
        from distributed import LocalCluster

        cluster = LocalCluster()

        return cluster.get_client()
    elif is_scheduler_address(cluster_config):
        from distributed import Client

        return Client(cluster_config)
    elif "/" in cluster_config:
        raise NotImplementedError("not yet supported")
    else:
        import dask_hpcconfig

        cluster = dask_hpcconfig.cluster(cluster_config)
        return cluster.get_client()


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
    client = connect_to_cluster(dask_cluster)

    decoded = decode_parameters(parameters)

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
