import warnings

import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
from shapely.errors import ShapelyDeprecationWarning


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

    (ax, ax2) = figure.subplots(
        ncols=2,
        subplot_kw={"projection": projection, "frameon": True},
    )

    ax.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
    ax.set_extent([x0, x1, y0, y1], crs=crs)
    gl1 = ax.gridlines(
        crs=crs,
        draw_labels=True,
        linewidth=0.6,
        color="gray",
        alpha=0.5,
        linestyle="-.",
    )
    gl1.xlabel_style = {"size": 15}
    gl1.ylabel_style = {"size": 15}
    gl1.right_labels = False
    gl1.top_labels = False

    cbar_kwargs = {
        "orientation": "horizontal",
        "shrink": 0.9,
        "fraction": 0.001,
        "pad": 0.05,
        "aspect": 80,
    }

    ds_["states"].plot(
        ax=ax,
        x="longitude",
        y="latitude",
        cbar_kwargs=cbar_kwargs | {"label": "State Probability"},
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
    gl2.xlabel_style = {"size": 15}
    gl2.ylabel_style = {"size": 15}
    gl2.left_labels = False
    gl2.top_labels = False

    ds_["emission"].plot(
        ax=ax2,
        x="longitude",
        y="latitude",
        cbar_kwargs=cbar_kwargs | {"label": "Emission Probability"},
        transform=ccrs.PlateCarree(),
        cmap="cool",
    )
    figure.suptitle(title)
    figure.subplots_adjust(top=0.9, wspace=-0.9)

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


def plot_trajectories(trajectories, *, subplots=False, **kwargs):
    import holoviews as hv

    if not subplots:
        return trajectories.hvplot(**kwargs)
    else:
        plots = [traj.hvplot(title=traj.id, **kwargs) for traj in trajectories]
        return hv.Layout(plots).cols(2)
