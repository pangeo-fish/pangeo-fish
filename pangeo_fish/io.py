import io
import os

import fsspec
import pandas as pd


def read_tag_database(url):
    """read the tag database

    Parameters
    ----------
    url : str, path-like or file-like
        The file path, url, path-like or pre-opened file object of the file to open

    Returns
    -------
    tag_database : pandas.DataFrame
        The opened tag database
    """
    return pd.read_csv(url, sep=";")


def read_detection_database(url):
    """read the detection database

    Parameters
    ----------
    url : str, path-like or file-like
        The file path, url, path-like or pre-opened file object of the file to open

    Returns
    -------
    detection_database : pandas.DataFrame
        The opened detection database

    """
    if isinstance(url, os.PathLike):
        url = os.fsspath(url)

    if isinstance(url, str):
        opened = fsspec.open(url, mode="r")
    else:
        opened = url

    with opened as f:
        # work around the weird quoting
        lines = (line.replace('"', "") for line in f)
        data = "\n".join(lines)

    content = io.StringIO(data)
    return (
        pd.read_csv(content, parse_dates=[1])
        .rename(columns={"date_time": "time"})
        .set_index("time")
    )
