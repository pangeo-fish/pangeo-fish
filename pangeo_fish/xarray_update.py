import xarray as xr


@xr.register_dataset_accessor("healpix")
class Accessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(self, var="pdf", refinement_level=8, alpha=0.8, ellipsoid="sphere"):
        """Return a interactive representation of healpix data contained on a xarray.

        Parameters
        ----------
        var : str
            the name of the variable you want to represent. This variable have to be in 'cells' coordinates
        refinement_level : int
            the level of resolution of healpix
        alpha : bool
            determine the transparence of the representation 0 is totally transparent, 1 is the max. Allows to see the chart
        ellipsoid : str
            The model for the healpix representation : "sphere" or "WGS84"
        Returns
        -------
        an interactive map
        """
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
