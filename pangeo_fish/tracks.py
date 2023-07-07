import movingpandas as mpd
from tlz.functoolz import curry, do, pipe

from .functoolz import lookup


def to_trajectory(ds, name, crs=None):
    return ds.to_pandas().pipe(
        mpd.Trajectory, traj_id=name, x="longitude", y="latitude"
    )


def additional_quantities(traj, quantities):
    quantity_methods = {
        "speed": curry(mpd.Trajectory.add_speed, name="speed", units=("km", "h")),
        "distance": curry(mpd.Trajectory.add_distance, name="distance", units="km"),
    }

    lookup_method = curry(lookup, quantity_methods, message="unknown quantity: {key}")
    funcs = [curry(do, lookup_method(quantity)) for quantity in quantities]

    return pipe(traj.copy(), *funcs)
