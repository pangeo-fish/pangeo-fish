import geopandas
import movingpandas


def to_geopandas(ds, cols=["longitude", "latitude"], crs=None):
    arr = geopandas.points_from_xy(*[ds[col] for col in cols], crs=crs)
    gdf = geopandas.GeoDataFrame({"geometry": arr}, index=ds.time)
    return gdf


def to_trajectory(ds, name, crs=None):
    gdf = to_geopandas(ds, cols=["longitude", "latitude"], crs=crs)
    return movingpandas.Trajectory(gdf, traj_id=name)
