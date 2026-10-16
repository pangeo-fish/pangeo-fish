import xarray as xr


@xr.register_dataset_accessor("healpix")
class Accessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(self, var="pdf", refinement_level=None, alpha=0.8, ellipsoid=None):
        """Return a interactive representation of healpix data contained on a xarray.

        Parameters
        ----------
        var : str
            the name of the variable you want to represent. This variable have to be in 'cells' coordinates
        refinement_level : int, optional
            the level of resolution of healpix. Defaults to the ``level`` stored
            in the DGGS attributes of ``cell_ids``.
        alpha : bool
            determine the transparence of the representation 0 is totally transparent, 1 is the max. Allows to see the chart
        ellipsoid : str, optional
            The model for the healpix representation : "sphere" or "WGS84".
            Defaults to the ``ellipsoid`` stored in the DGGS attributes of
            ``cell_ids`` (``"sphere"`` if absent).
        Returns
        -------
        an interactive map
        """
        attrs = self._obj["cell_ids"].attrs
        if "level" not in attrs:
            # the grid definition may only live in the xdggs index
            try:
                attrs = attrs | self._obj.dggs.grid_info.to_dict()
            except Exception:
                pass
        if refinement_level is None:
            refinement_level = int(attrs["level"])
        if ellipsoid is None:
            ellipsoid = attrs.get("ellipsoid") or "sphere"

        plot = (
            self._obj[var]
            .compute()
            .dggs.decode(
                {
                    "grid_name": "healpix",
                    "level": refinement_level,
                    "indexing_scheme": attrs.get("indexing_scheme", "nested"),
                    "ellipsoid": ellipsoid,
                }
            )
            .dggs.explore(alpha=alpha, cmap="viridis")
        )
        return plot
