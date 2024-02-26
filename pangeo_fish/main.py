import pathlib

import pint_xarray
import rich_click as click

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
def estimate():
    pass


@main.command("decode", short_help="produce the model output")
def decode():
    pass
