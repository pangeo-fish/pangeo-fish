import warnings

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.ticker as mticker
import numpy as np
from shapely.errors import ShapelyDeprecationWarning


def filter_by_states(ds):
    return ds.where(ds["states"].sum(dim="time", skipna=True).compute() > 0, drop=True)


def create_frame(ds, figure, index, *args, **kwargs):
    warnings.filterwarnings(
        action="ignore",
        category=ShapelyDeprecationWarning,  # in cartopy
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=r"No `(vmin|vmax)` provided. Data limits are calculated from input. Depending on the input this can take long. Pass `\1` to avoid this step",
    )

    ds_ = ds.drop_vars(["resolution"], errors="ignore").isel(time=index, drop=True)
    title = (
        f"time = {np.datetime_as_string(ds['time'].isel(time=index).data, unit='s')}"
    )

    projection = ccrs.Mercator()
    crs = ccrs.PlateCarree()
    bbox = (
        ds_["longitude"].min(),
        ds_["latitude"].min(),
        ds_["longitude"].max(),
        ds_["latitude"].max(),
    )
    x0, y0, x1, y1 = bbox
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    cbar_kwargs = {
        "orientation": "horizontal",
        "shrink": 0.65,
        "pad": 0.05,
        "aspect": 50,
        "format": formatter,
    }

    gs = figure.add_gridspec(nrows=1, ncols=2, hspace=0, wspace=-0.2, top=0.9)
    (ax1, ax2) = gs.subplots(
        subplot_kw={"projection": projection, "frameon": True},
        sharex=True,
        sharey=True,
    )

    ds_["states"].plot(
        ax=ax1,
        x="longitude",
        y="latitude",
        cbar_kwargs=cbar_kwargs | {"label": "State Probability"},
        transform=ccrs.PlateCarree(),
        cmap="cool",
    )

    ax1.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
    ax1.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
    ax1.set_extent([x0, x1, y0, y1], crs=crs)
    gl1 = ax1.gridlines(
        crs=crs,
        draw_labels=True,
        linewidth=0.6,
        color="gray",
        alpha=0.5,
        linestyle="-.",
    )
    # gl1.xlabel_style = {"size": 15}
    # gl1.ylabel_style = {"size": 15}
    gl1.right_labels = False
    gl1.top_labels = False

    ds_["emission"].plot(
        ax=ax2,
        x="longitude",
        y="latitude",
        cbar_kwargs=cbar_kwargs | {"label": "Emission Probability"},
        transform=ccrs.PlateCarree(),
        cmap="cool",
    )
    ax2.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
    ax2.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
    ax2.set_extent([x0, x1, y0, y1], crs=crs)

    gl2 = ax2.gridlines(
        crs=crs,
        draw_labels=True,
        linewidth=0.6,
        color="gray",
        alpha=0.5,
        linestyle="-.",
    )
    # gl2.xlabel_style = {"size": 15}
    # gl2.ylabel_style = {"size": 15}
    gl2.left_labels = False
    gl2.top_labels = False

    figure.suptitle(title)

    return None, None


def plot_map(
    arr,
    x="longitude",
    y="latitude",
    rasterize=True,
    geo=True,
    coastline="10m",
    tiles=None,
    cmap="cmo.amp",
    **kwargs,
):
    """wrapper around `DataArray.hvplot.quadmesh`, with different defaults"""
    return arr.hvplot.quadmesh(
        x=x,
        y=y,
        rasterize=rasterize,
        geo=geo,
        coastline=coastline,
        tiles=tiles,
        cmap=cmap,
        **kwargs,
    )
