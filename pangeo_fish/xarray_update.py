import xarray as xr


@xr.register_dataset_accessor("pdf_plot")
class Accessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot_healpix(
        self, var="pdf", refinement_level=8, alpha=0.8, ellipsoid="sphere"
    ):
        plot = (
            self._obj[var]
            .compute()
            .dggs.decode(
                {
                    "grid_name": "healpix",
                    "level": refinement_level,
                    "indexing_scheme": "nested",
                    "ellipsoid": ellipsoid,
                }
            )
            .dggs.explore(alpha=alpha, cmap="viridis")
        )
        return plot
