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
