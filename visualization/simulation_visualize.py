import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.path as mpl_path
import matplotlib.patches as mpl_patches
import matplotlib.collections as mpl_coll
import shapely.geometry as sp
import skgeom as sg
import numpy as np
from typing import Union, List
import visualization.sdd_visualize as sdd_visualize


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
    else:
        print(f"incorrect object type:\n{type(poly)}")
        return None
    
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
    [plot_sg_polygon(ax, poly, edgecolor="red", facecolor="red", alpha=0.1)
     for poly in no_ego_buffers.union(frame_box).polygons]
    [plot_sg_polygon(ax, poly, edgecolor="red", facecolor="red", hatch="\\\\\\", alpha=0.2)
     for poly in no_ego_wedges.polygons]
    [plot_sg_polygon(ax, poly, edgecolor="cyan", facecolor="cyan", hatch="///", alpha=0.2)
     for poly in target_agents_fully_observable_regions.polygons]
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
    [ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c="purple") for p1, p2 in zip(p1s, p2s)]
    [plot_sg_polygon(ax, poly, edgecolor="red", facecolor="red", alpha=0.2) for poly in occluded_regions.polygons]


def visualize_occlusion_simulation(instance_dict: dict, simulation_dict: dict) -> None:
    agents = instance_dict["agents"]
    past_window = instance_dict["past_window"]
    future_window = instance_dict["future_window"]
    full_window = np.concatenate((past_window, future_window))

    target_agent_indices = simulation_dict["target_agent_indices"]
    occlusion_windows = simulation_dict["occlusion_windows"]
    frame_box = simulation_dict["frame_box"]
    agent_visipoly_buffers = simulation_dict["agent_visipoly_buffers"]
    no_occluder_buffers = simulation_dict["no_occluder_buffers"]
    no_ego_buffers = simulation_dict["no_ego_buffers"]
    no_ego_wedges = simulation_dict["no_ego_wedges"]
    targets_fullobs_regions = simulation_dict["targets_fullobs_regions"]
    yes_ego_triangles = simulation_dict["yes_ego_triangles"]
    ego_point = simulation_dict["ego_point"]
    ego_buffer = simulation_dict["ego_buffer"]
    p1_area = simulation_dict["p1_area"]
    p1_triangles = simulation_dict["p1_triangles"]
    p1_visipolys = simulation_dict["p1_visipolys"]
    p2_area = simulation_dict["p2_area"]
    p2_triangles = simulation_dict["p2_triangles"]
    occluders = simulation_dict["occluders"]
    occluded_regions = simulation_dict["occluded_regions"]

    # visualization part
    p_occls = [agents[idx].position_at_timestep(full_window[occlusion_window[0]])
               for idx, occlusion_window in zip(target_agent_indices, occlusion_windows)]
    p_disoccls = [agents[idx].position_at_timestep(full_window[occlusion_window[1]])
                  for idx, occlusion_window in zip(target_agent_indices, occlusion_windows)]
    p1s = [occluder[0] for occluder in occluders]
    p2s = [occluder[1] for occluder in occluders]

    fig, axs = plt.subplots(nrows=2, ncols=3)
    [sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict) for ax in axs.reshape(-1)]
    plot_simulation_step_1(axs[0, 0], agent_visipoly_buffers, no_occluder_buffers, no_ego_buffers, frame_box)
    plot_simulation_step_2(axs[0, 1], agent_visipoly_buffers, no_ego_buffers, frame_box, no_ego_wedges,
                           targets_fullobs_regions, p_occls, p_disoccls)
    plot_simulation_step_3(axs[0, 2], yes_ego_triangles, p_occls, p_disoccls, ego_point, no_occluder_buffers,
                           ego_buffer, p1_area)
    plot_simulation_step_4(axs[1, 0], p_occls, p_disoccls, ego_point, p1_triangles, p1s, p1_visipolys, p2_area)
    plot_simulation_step_5(axs[1, 1], p_occls, p_disoccls, ego_point, p2_triangles, p1s, p2s)
    plot_simulation_step_6(axs[1, 2], p_occls, p_disoccls, ego_point, p1s, p2s, occluded_regions)

    # I. agent buffers & frame_box
    fig1, ax1 = plt.subplots()
    sdd_visualize.visualize_training_instance(ax1, instance_dict=instance_dict)
    plot_simulation_step_1(ax1, agent_visipoly_buffers, no_occluder_buffers, no_ego_buffers, frame_box)

    # II. target agents' occlusion timesteps, wedges and visibility polygons
    fig2, ax2 = plt.subplots()
    sdd_visualize.visualize_training_instance(ax2, instance_dict=instance_dict)
    plot_simulation_step_2(ax2, agent_visipoly_buffers, no_ego_buffers, frame_box, no_ego_wedges,
                           targets_fullobs_regions, p_occls, p_disoccls)

    # III. triangulated ego regions, ego point, ego buffer, p1_ego_traj triangles
    fig3, ax3 = plt.subplots()
    sdd_visualize.visualize_training_instance(ax3, instance_dict=instance_dict)
    plot_simulation_step_3(ax3, yes_ego_triangles, p_occls, p_disoccls, ego_point, no_occluder_buffers, ego_buffer,
                           p1_area)

    # IV. triangulated p1_regions, p1, p1 visibility polygon, p2_ego_traj triangles
    fig4, ax4 = plt.subplots()
    sdd_visualize.visualize_training_instance(ax4, instance_dict=instance_dict)
    plot_simulation_step_4(ax4, p_occls, p_disoccls, ego_point, p1_triangles, p1s, p1_visipolys, p2_area)

    # V. triangulated p2_regions, p2
    fig5, ax5 = plt.subplots()
    sdd_visualize.visualize_training_instance(ax5, instance_dict=instance_dict)
    plot_simulation_step_5(ax5, p_occls, p_disoccls, ego_point, p2_triangles, p1s, p2s)

    # VI. occluder, ego_point visibility
    fig6, ax6 = plt.subplots()
    sdd_visualize.visualize_training_instance(ax6, instance_dict=instance_dict)
    plot_simulation_step_6(ax6, p_occls, p_disoccls, ego_point, p1s, p2s, occluded_regions)

    plt.show()

