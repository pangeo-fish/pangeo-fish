import movingpandas as mpd
from tlz.functoolz import curry


def to_trajectory(ds, name, crs=None):
    return ds.to_pandas().pipe(
        mpd.Trajectory, traj_id=name, x="longitude", y="latitude"
    )


def additional_quantities(traj, quantities):
    quantity_methods = {
        "speed": curry(mpd.Trajectory.add_speed, name="speed", units=("km", "h")),
        "distance": curry(mpd.Trajectory.add_distance, name="distance", units="km"),
    }

    extended = traj.copy()

    for quantity in quantities:
        method = quantity_methods.get(quantity)
        if method is None:
            raise ValueError(f"unknown quantity: {quantity}")

        method(extended)

    return extended
