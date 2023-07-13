import matplotlib.axes
import matplotlib.path as mpl_path
import matplotlib.patches as mpl_patches
import matplotlib.collections as mpl_coll
import shapely.geometry as sp
import skgeom as sg
import numpy as np
from typing import Union


def plot_sp_polygon(ax: matplotlib.axes.Axes, poly: sp.Polygon, **kwargs) -> None:
    path = mpl_path.Path.make_compound_path(
        mpl_path.Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[mpl_path.Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors]
    )

    patch = mpl_patches.PathPatch(path, **kwargs)
    collection = mpl_coll.PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()


def plot_sg_polygon(ax: matplotlib.axes.Axes, poly: Union[sg.Polygon, sg.PolygonWithHoles], **kwargs) -> None:
    if isinstance(poly, sg.Polygon):
        path = mpl_path.Path(poly.coords)
    elif isinstance(poly, sg.PolygonWithHoles):
        path = mpl_path.Path.make_compound_path(
            mpl_path.Path(poly.outer_boundary().coords),
            *[mpl_path.Path(hole.coords) for hole in poly.holes]
        )
    else:
        raise TypeError(f"incorrect object type:\n{type(poly)}")

    patch = mpl_patches.PathPatch(path, **kwargs)
    collection = mpl_coll.PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()