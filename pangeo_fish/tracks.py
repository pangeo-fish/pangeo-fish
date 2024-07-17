import operator

import movingpandas as mpd
from tlz.functoolz import curry, do, pipe

from pangeo_fish.functoolz import lookup


def to_trajectory(ds, name, crs=None):
    return ds.to_pandas().pipe(
        mpd.Trajectory, traj_id=name, x="longitude", y="latitude"
    )


def additional_quantities(traj, quantities):
    if not quantities:
        return traj

    quantity_methods = {
        "speed": operator.methodcaller("add_speed", name="speed", units=("km", "h")),
        "distance": operator.methodcaller("add_distance", name="distance", units="km"),
    }

    lookup_method = curry(lookup, quantity_methods, message="unknown quantity: {key}")
    funcs = [curry(do, lookup_method(quantity)) for quantity in quantities]

    return pipe(traj.copy(), *funcs)


def to_dataframe(gdf):
    coords = gdf.geometry.get_coordinates().rename(
        columns={"x": "longitude", "y": "latitude"}
    )
    return gdf.merge(coords, left_index=True, right_index=True).drop(
        columns=["geometry", "traj_id"]
    )
