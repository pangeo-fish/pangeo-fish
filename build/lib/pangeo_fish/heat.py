from xarray_healpy.conversions import geographic_to_cartesian
from pangeo_fish.healpy import (
    astronomic_to_cartesian,
    astronomic_to_cell_ids,
    geographic_to_astronomic,
)
from xarray_healpy.operations import buffer_points
import xarray as xr


def powerpalnt_emission_map(pp_map, emission, buffer_size, rot):
    grid = emission[["time", "cell_ids", "mask"]].compute()
    cell_ids = grid["cell_ids"]

    positions = geographic_to_cartesian(
        lon=pp_map["Longitude"],
        lat=pp_map["Latitude"],
        rot=rot,
    )

    masks = buffer_points(
        cell_ids,
        positions,
        buffer_size=buffer_size.m_as("m"),
        nside=2 ** cell_ids.attrs["level"],
        factor=2**16,
        intersect=True,
    ).drop_vars(["cell_ids"])
    combined_masks = masks.sum(dim="index")
    return (
        combined_masks.transpose("y", "x")
        .assign_attrs({"buffer_size": buffer_size.m_as("m")})
        .where(grid["mask"])
    )


def heat_regulation(emission, detections, combined_masks):
    for time in emission["time"]:
        if detections["predicted_label"].loc[time] == 1.0:
            emission["pdf"].loc[time] = combined_masks
    return emission
