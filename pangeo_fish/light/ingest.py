"""Tag CSV ingestion helpers.

Converts raw manufacturer files into the standard folder structure expected
by :func:`pangeo_fish.io.open_tag`:

.. code-block:: text

    {output_dir}/{tag_name}/
        dst.csv              ← time, temperature, pressure, light
        tagging_event.csv   ← copied from the source file
        metadata.json        ← tag_name, tag_type

Supported tag types
-------------------
``"dst"``
    Standard pangeo-fish format — already-processed ``dst.csv`` with
    columns ``time``, ``temperature``, ``pressure``, ``light``.

``"lotek"``
    Lotek LAT2810 — semicolon-separated, comma decimal,
    timestamp format ``%H:%M:%S %d/%m/%y``.

``"wc_psat"``
    Wildlife Computers MiniPAT PSAT — 10-minute Series CSV
    (``*-Series.csv``) with columns ``Day``, ``Time``, ``Depth``,
    ``Temperature``.  No raw light counts are available, so ``light``
    is set to NaN and ``HAS_LIGHT`` must be ``False``.
"""

import json
import os
import shutil

import numpy as np
import pandas as pd


def load_tag_csv(path, tag_type):
    """Load a raw manufacturer CSV and return a standardised DataFrame.

    Parameters
    ----------
    path : str or path-like
        Path to the CSV file.
    tag_type : {"lotek", "wc_psat"}
        Manufacturer / format identifier.

    Returns
    -------
    pd.DataFrame
        Index ``time`` (UTC-naive), columns
        ``temperature``, ``pressure``, ``light``.

    Raises
    ------
    ValueError
        If *tag_type* is not recognised.
    """
    tag_type = tag_type.lower()

    if tag_type == "lotek":
        df_raw = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            names=[
                "TimeS",
                "LightIntensity",
                "ExtTemp",
                "Pressure",
                "C_TooDimFlag",
                "_",
            ],
            skiprows=1,
            usecols=[0, 1, 2, 3],
        )
        df_raw = df_raw.dropna(subset=["TimeS"])
        df_raw["time"] = pd.to_datetime(
            df_raw["TimeS"].str.strip(), format="%H:%M:%S %d/%m/%y", errors="coerce"
        )
        df_raw = df_raw.dropna(subset=["time"]).set_index("time").sort_index()
        dst = df_raw.rename(
            columns={
                "ExtTemp": "temperature",
                "Pressure": "pressure",
                "LightIntensity": "light",
            }
        )[["temperature", "pressure", "light"]]

    elif tag_type == "wc_psat":
        df_raw = pd.read_csv(path)
        df_raw["time"] = pd.to_datetime(
            df_raw["Day"] + " " + df_raw["Time"], format="%d-%b-%Y %H:%M:%S"
        )
        df_raw = (
            df_raw.dropna(subset=["Depth", "Temperature"])
            .sort_values("time")
            .set_index("time")
        )
        dst = pd.DataFrame(
            {
                "temperature": df_raw["Temperature"],
                "pressure": df_raw["Depth"],
                "light": np.nan,
            }
        )

    elif tag_type == "dst":
        # Standard pangeo-fish dst.csv format — already processed
        dst = pd.read_csv(path, parse_dates=["time"], index_col="time").sort_index()
        dst = dst[["temperature", "pressure", "light"]]

    else:
        raise ValueError(
            f"Unknown tag_type {tag_type!r}. Supported: 'lotek', 'wc_psat', 'dst'."
        )

    print(f"  {len(dst):,} rows | {dst.index.min()} → {dst.index.max()}")
    return dst


def prepare_tag_folder(
    raw_csv_path,
    tag_type,
    tagging_events_path,
    output_dir,
    tag_name,
):
    """Parse a raw manufacturer CSV and write the standard tag folder.

    Creates ``{output_dir}/{tag_name}/`` containing:

    * ``dst.csv`` — standardised time-series (time, temperature, pressure, light)
    * ``tagging_event.csv`` — copied from *tagging_events_path*
    * ``metadata.json`` — ``{"tag_name": ..., "tag_type": ...}``

    The resulting folder can be opened with :func:`pangeo_fish.io.open_tag`.

    Parameters
    ----------
    raw_csv_path : str or path-like
        Path to the raw manufacturer CSV file.
    tag_type : {"lotek", "wc_psat"}
        Manufacturer / format identifier passed to :func:`load_tag_csv`.
    tagging_events_path : str or path-like
        Path to the ``tagging_event.csv`` file with columns
        ``event_name``, ``time``, ``longitude``, ``latitude``.
    output_dir : str or path-like
        Root directory where the tag folder will be created.
    tag_name : str
        Tag identifier — used as the subfolder name and stored in metadata.

    Returns
    -------
    folder : str
        Absolute path to the created ``{output_dir}/{tag_name}/`` folder.
    """
    raw_csv_path = os.path.expanduser(str(raw_csv_path))
    tagging_events_path = os.path.expanduser(str(tagging_events_path))
    output_dir = os.path.expanduser(str(output_dir))

    dst = load_tag_csv(raw_csv_path, tag_type)

    folder = os.path.join(output_dir, tag_name)
    os.makedirs(folder, exist_ok=True)

    # dst.csv
    dst.to_csv(os.path.join(folder, "dst.csv"), date_format="%Y-%m-%dT%H:%M:%S")

    # tagging_event.csv
    shutil.copy(tagging_events_path, os.path.join(folder, "tagging_event.csv"))

    # metadata.json
    metadata = {"tag_name": tag_name, "tag_type": tag_type}
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Tag folder ready: {folder}")
    return folder
