import cf_xarray  # noqa: F401
import numpy as np


def marc_sigma_to_depth(model):
    s = model.cf["ocean_s_coordinate"]
    eta = model.cf["sea_surface_height_above_sea_level"]
    depth = model.cf["model_sea_floor_depth_below_sea_level"]
    a = model.cf["ocean_s_coordinate_surface_control"]
    b = model.cf["ocean_s_coordinate_bottom_control"]
    depth_c = model.cf["ocean_s_coordinate_thick_of_surface_resolution"]

    C = (1.0 - b) * np.sinh(a * s) / np.sinh(a) + b * (
        np.tanh(a * (s + 0.5)) - np.tanh(0.5 * a)
    ) / (2.0 * np.tanh(0.5 * a))

    z = (eta * (1.0 + s) + depth_c * s + (depth - depth_c) * C).astype("float32")

    depth_ = (z + eta).assign_attrs({"units": "m"})
    bottom = (eta + depth).assign_attrs({"units": "m"})

    return model.assign(depth=depth_, bottom=bottom)[["TEMP", "depth", "bottom", "H0"]]
