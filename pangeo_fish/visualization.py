import warnings

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cf
import cmocean  # noqa: F401
import hvplot.xarray  # noqa: F401
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

    default_xlim = [ds_["longitude"].min(), ds_["longitude"].max()]
    default_ylim = [ds_["latitude"].min(), ds_["latitude"].max()]

    x0, x1 = kwargs.get("xlim", default_xlim)
    y0, y1 = kwargs.get("ylim", default_ylim)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar_kwargs = {
        "orientation": "horizontal",
        "shrink": 0.65,
        "pad": 0.05,
        "aspect": 50,
        "format": formatter,
        "use_gridspec": True,
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
        xlim=[x0, x1],
        ylim=[y0, y1],
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
    gl1.right_labels = False
    gl1.top_labels = False

    ds_["emission"].plot(
        ax=ax2,
        x="longitude",
        y="latitude",
        cbar_kwargs=cbar_kwargs | {"label": "Emission Probability"},
        transform=ccrs.PlateCarree(),
        cmap="cool",
        xlim=[x0, x1],
        ylim=[y0, y1],
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
    gl2.left_labels = False
    gl2.top_labels = False

    figure.suptitle(title)

    return None, None


def render_frame(ds, *args, **kwargs):
    """
        .. warning::
            To use with `dask.map_blocks()`
            `ds` must have the variable "time_index", representing the time index.

     Parameters
        ----------
        output : str, "."
            Name of the folder to save the frame
    """
    def _render_frame(ds, figure, **kwargs):
        """render a frame

        Parameters
        ----------
        vmax : dict[str, float], default to {"states": None, "emission": None}
            Mpping of the vmax values for the pdfs `states` and `emission`

        Returns
        -------
        fig : Figure
            The plt.Figure
        """
        time = ds["time"].values[0]
        ds_ = ds.drop_vars(["resolution"], errors="ignore")
        title = (
            f"Time = {np.datetime_as_string(time, unit="s")}"
        )
        projection = ccrs.Mercator()
        crs = ccrs.PlateCarree()

        default_xlim = [ds_["longitude"].min(), ds_["longitude"].max()]
        default_ylim = [ds_["latitude"].min(), ds_["latitude"].max()]

        x0, x1 = kwargs.get("xlim", default_xlim)
        y0, y1 = kwargs.get("ylim", default_ylim)
        vmax = kwargs.get("vmax", {})

        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))

        cbar_kwargs = {
            "orientation": "horizontal",
            "shrink": 0.65,
            "pad": 0.05,
            "aspect": 50,
            "format": formatter,
            "use_gridspec": True,
            "extend": "max"
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
            xlim=[x0, x1],
            ylim=[y0, y1],
            vmin=0,
            vmax=vmax.get("states", None)
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
        gl1.right_labels = False
        gl1.top_labels = False

        ds_["emission"].plot(
            ax=ax2,
            x="longitude",
            y="latitude",
            cbar_kwargs=cbar_kwargs | {"label": "Emission Probability"},
            transform=ccrs.PlateCarree(),
            cmap="cool",
            xlim=[x0, x1],
            ylim=[y0, y1],
            vmin=0,
            vmax=vmax.get("emission", None)
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
        gl2.left_labels = False
        gl2.top_labels = False

        figure.suptitle(title)
        ax1.set_title("")
        ax2.set_title("")
        return figure

    figure = plt.figure(figsize=(14, 8)) # figsize=(12, 6)

    try:
        _render_frame(ds, figure, kwargs)
        time_index = ds["time_index"].values[0]
        frame_dir = kwargs.get("output", ".")
        figure.savefig(f"{frame_dir}/frame_{time_index:05d}.png")#, bbox_inches="tight", pad_inches=0.2)
    except Exception as e:
        print(f"============ Exception at time {ds["time_index"].values[0]} ==============")
        print(e)
        print("=========================================================")
    finally:
        plt.close(figure)

    return ds


def plot_map(
    arr,
    bbox,
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
        xlim=bbox["longitude"],
        ylim=bbox["latitude"],
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


# def create_frames()