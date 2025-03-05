import pathlib


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
