import matplotlib.axes
import shapely.geometry as sp
import skgeom as sg
import matplotlib.path as mpl_path
import matplotlib.patches as mpl_patches
import matplotlib.collections as mpl_coll
import numpy as np
from typing import Union, List


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
        # coords = np.concatenate([poly.coords, [poly.coords[-1]]])
        # path = mpl_path.Path(coords)
    elif isinstance(poly, sg.PolygonWithHoles):
        path = mpl_path.Path.make_compound_path(
            mpl_path.Path(poly.outer_boundary().coords),
            *[mpl_path.Path(hole.coords) for hole in poly.holes]
        )

    patch = mpl_patches.PathPatch(path, **kwargs)
    collection = mpl_coll.PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()


def plot_simulation_step_1(
        ax: matplotlib.axes.Axes,
        agent_visipoly_buffers: List[sg.Polygon],
        no_occluder_buffers: List[sg.Polygon],
        no_ego_buffers: sg.PolygonSet,
        frame_box: sg.PolygonWithHoles
) -> None:
    # I. agent buffers & frame_box
    [plot_sg_polygon(ax, poly, edgecolor="blue", facecolor="blue", alpha=0.2) for poly in agent_visipoly_buffers]
    [plot_sg_polygon(ax, poly, edgecolor="orange", facecolor="orange", alpha=0.2) for poly in no_occluder_buffers]
    [plot_sg_polygon(ax, poly, edgecolor="red", facecolor="red", alpha=0.2) for poly in no_ego_buffers.polygons]
    plot_sg_polygon(ax, frame_box, edgecolor="red", facecolor="red", alpha=0.2)


def plot_simulation_step_2(
        ax: matplotlib.axes.Axes,
        agent_visipoly_buffers: List[sg.Polygon],
        no_ego_buffers: sg.PolygonSet,
        frame_box: sg.PolygonWithHoles,
        no_ego_wedges: sg.PolygonSet,
        target_agents_fully_observable_regions: sg.PolygonSet,
        p_occls: List[np.array],
        p_disoccls: List[np.array]
) -> None:
    # II. target agents' occlusion timesteps, wedges and visibility polygons
    [plot_sg_polygon(ax, poly, edgecolor="blue", facecolor="blue", alpha=0.1) for poly in agent_visipoly_buffers]
    [plot_sg_polygon(ax, poly, edgecolor="red", facecolor="red", alpha=0.1) for poly in no_ego_buffers.union(frame_box).polygons]
    [plot_sg_polygon(ax, poly, edgecolor="red", facecolor="red", hatch="\\\\\\", alpha=0.2) for poly in no_ego_wedges.polygons]
    [plot_sg_polygon(ax, poly, edgecolor="cyan", facecolor="cyan", hatch="///", alpha=0.2) for poly in target_agents_fully_observable_regions.polygons]
    [ax.scatter(*p_occl, marker="x", c="Yellow") for p_occl in p_occls]
    [ax.scatter(*p_disoccl, marker="x", c="Yellow") for p_disoccl in p_disoccls]


def plot_simulation_step_3(
        ax: matplotlib.axes.Axes,
        yes_triangles: List[sg.Polygon],
        p_occls: List[np.array],
        p_disoccls: List[np.array],
        ego_point: np.array,
        no_occluder_buffers: List[sg.Polygon],
        ego_buffer: sg.Polygon,
        p1_ego_traj_triangles: List[sg.Polygon]
) -> None:
    # III. triangulated ego regions, ego point, ego buffer, p1_ego_traj triangles
    [plot_sg_polygon(ax, poly, edgecolor="green", facecolor="green", alpha=0.2) for poly in yes_triangles]
    [ax.scatter(*p_occl, marker="x", c="Yellow") for p_occl in p_occls]
    [ax.scatter(*p_disoccl, marker="x", c="Yellow") for p_disoccl in p_disoccls]
    ax.scatter(*ego_point, marker="x", c="red")
    [plot_sg_polygon(ax, poly, edgecolor="orange", facecolor="orange", alpha=0.2) for poly in no_occluder_buffers]
    plot_sg_polygon(ax, ego_buffer, edgecolor="orange", facecolor="orange", alpha=0.2)
    [plot_sg_polygon(ax, poly, edgecolor="yellow", facecolor="yellow", alpha=0.2) for poly in p1_ego_traj_triangles]


def plot_simulation_step_4(
        ax: matplotlib.axes.Axes,
        p_occls: List[np.array],
        p_disoccls: List[np.array],
        ego_point: np.array,
        triangulated_p1_regions: List[sg.Polygon],
        p1s: List[np.array],
        p1_visipolys: List[sg.Polygon],
        p2_ego_traj_triangles: List[sg.Polygon]
) -> None:
    # IV. triangulated p1_regions, p1, p1 visibility polygon, p2_ego_traj triangles
    [ax.scatter(*p_occl, marker="x", c="Yellow") for p_occl in p_occls]
    [ax.scatter(*p_disoccl, marker="x", c="Yellow") for p_disoccl in p_disoccls]
    ax.scatter(*ego_point, marker="x", c="red")
    [plot_sg_polygon(ax, poly, edgecolor="yellow", facecolor="yellow", alpha=0.2) for poly in triangulated_p1_regions]
    [ax.scatter(*point, marker="x", c="purple") for point in p1s]
    [plot_sg_polygon(ax, poly, edgecolor="cyan", facecolor="cyan", alpha=0.2) for poly in p1_visipolys]
    [plot_sg_polygon(ax, poly, edgecolor="yellow", facecolor="yellow", alpha=0.2) for poly in p2_ego_traj_triangles]


def plot_simulation_step_5(
        ax: matplotlib.axes.Axes,
        p_occls: List[np.array],
        p_disoccls: List[np.array],
        ego_point: np.array,
        triangulated_p2_regions: List[sg.Polygon],
        p1s: List[np.array],
        p2s: List[np.array],
) -> None:
    # V. triangulated p2_regions, p2
    [ax.scatter(*p_occl, marker="x", c="Yellow") for p_occl in p_occls]
    [ax.scatter(*p_disoccl, marker="x", c="Yellow") for p_disoccl in p_disoccls]
    ax.scatter(*ego_point, marker="x", c="red")
    [plot_sg_polygon(ax, poly, edgecolor="yellow", facecolor="yellow", alpha=0.2) for poly in triangulated_p2_regions]
    [ax.scatter(point[0], point[1], marker="x", c="purple") for point in p1s]
    [ax.scatter(point[0], point[1], marker="x", c="purple") for point in p2s]


def plot_simulation_step_6(
        ax: matplotlib.axes.Axes,
        p_occls: List[np.array],
        p_disoccls: List[np.array],
        ego_point: np.array,
        p1s: List[np.array],
        p2s: List[np.array],
        occluded_regions: sg.PolygonSet
) -> None:
    # VI. occluder, ego_point visibility
    [ax.scatter(*p_occl, marker="x", c="Yellow") for p_occl in p_occls]
    [ax.scatter(*p_disoccl, marker="x", c="Yellow") for p_disoccl in p_disoccls]
    ax.scatter(*ego_point, marker="x", c="red")
    [ax.scatter(point[0], point[1], marker="x", c="purple") for point in p1s]
    [ax.scatter(point[0], point[1], marker="x", c="purple") for point in p2s]
    [plot_sg_polygon(ax, poly, edgecolor="red", facecolor="red", alpha=0.2) for poly in occluded_regions.polygons]
