from importlib.metadata import version

try:
    __version__ = version("pangeo-fish")
except Exception:
    __version__ = "9999"
