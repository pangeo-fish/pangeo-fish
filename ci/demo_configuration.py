"""Read serializable Papermill defaults without importing the scientific stack."""

import ast


def demo_settings(notebook, tag_name="A19124"):
    if tag_name not in {"A19124", "A18828"}:
        raise ValueError("Unsupported public demo tag")
    cells = [
        c
        for c in notebook["cells"]
        if "parameters" in c.get("metadata", {}).get("tags", [])
    ]
    if len(cells) != 1:
        raise ValueError("Expected exactly one Papermill parameters cell")
    values = {}
    for node in ast.parse("".join(cells[0]["source"])).body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
        ):
            raise ValueError("Papermill defaults must be simple literal assignments")
        values[node.targets[0].id] = ast.literal_eval(node.value)
    parameters = {"tag_name": tag_name}
    return values | parameters, parameters


# Where the demo tag files are read from. "eosc-minio" is the public read-only
# mirror of the data-taos demo tags on the Pangeo@EOSC MinIO. It is the default
# because data-taos.ifremer.fr is often unreachable. Only the tag files are
# mirrored: the complete demo still reads bathymetry and tides from data-taos.
#
# To switch the default back, swap the two lines below. The manual run
# (workflow_dispatch) has its own default in
# .github/workflows/demo-notebooks.yaml.
DEFAULT_TAG_SOURCE = "eosc-minio"
# DEFAULT_TAG_SOURCE = "data-taos"

TAG_SOURCES = {
    "data-taos": {},
    "eosc-minio": {
        "tag_root": "s3://todaka-pangeo-fish-demo/tag",
        "tag_storage_options": {
            "anon": True,
            "client_kwargs": {
                "endpoint_url": "https://pangeo-eosc-minioapi.vm.fedcloud.eu"
            },
        },
    },
}


def tag_source_parameters(source=None):
    source = source or DEFAULT_TAG_SOURCE
    if source not in TAG_SOURCES:
        raise ValueError(f"Unknown tag source {source!r}")
    return dict(TAG_SOURCES[source])
