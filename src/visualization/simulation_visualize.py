import matplotlib.pyplot as plt
import matplotlib.axes
import skgeom as sg
import numpy as np
from typing import List
import src.visualization.sdd_visualize as sdd_visualize
from src.data.sdd_dataloader import StanfordDroneDataset
from src.occlusion_simulation.simple_occlusion import simulate_occlusions
from src.visualization.plot_utils import plot_sg_polygon


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
    occlusion_target_coords = simulation_dict["occlusion_target_coords"]
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
    p_occls = [coords[0] for coords in occlusion_target_coords]
    p_disoccls = [coords[1] for coords in occlusion_target_coords]
    p1s = [occluder[0] for occluder in occluders]
    p2s = [occluder[1] for occluder in occluders]

    # I. agent buffers & frame_box
    fig1, ax1 = plt.subplots()
    sdd_visualize.draw_map_numpy(draw_ax=ax1, scene_image=instance_dict["scene_image"])
    sdd_visualize.visualize_training_instance(ax1, instance_dict=instance_dict)
    plot_simulation_step_1(ax1, agent_visipoly_buffers, no_occluder_buffers, no_ego_buffers, frame_box)

    # II. target agents' occlusion timesteps, wedges and visibility polygons
    fig2, ax2 = plt.subplots()
    sdd_visualize.draw_map_numpy(draw_ax=ax2, scene_image=instance_dict["scene_image"])
    sdd_visualize.visualize_training_instance(ax2, instance_dict=instance_dict)
    plot_simulation_step_2(ax2, agent_visipoly_buffers, no_ego_buffers, frame_box, no_ego_wedges,
                           targets_fullobs_regions, p_occls, p_disoccls)

    # III. triangulated ego regions, ego point, ego buffer, p1_ego_traj triangles
    fig3, ax3 = plt.subplots()
    sdd_visualize.draw_map_numpy(draw_ax=ax3, scene_image=instance_dict["scene_image"])
    sdd_visualize.visualize_training_instance(ax3, instance_dict=instance_dict)
    plot_simulation_step_3(ax3, yes_ego_triangles, p_occls, p_disoccls, ego_point, no_occluder_buffers, ego_buffer,
                           p1_area)

    # IV. triangulated p1_regions, p1, p1 visibility polygon, p2_ego_traj triangles
    fig4, ax4 = plt.subplots()
    sdd_visualize.draw_map_numpy(draw_ax=ax4, scene_image=instance_dict["scene_image"])
    sdd_visualize.visualize_training_instance(ax4, instance_dict=instance_dict)
    plot_simulation_step_4(ax4, p_occls, p_disoccls, ego_point, p1_triangles, p1s, p1_visipolys, p2_area)

    # V. triangulated p2_regions, p2
    fig5, ax5 = plt.subplots()
    sdd_visualize.draw_map_numpy(draw_ax=ax5, scene_image=instance_dict["scene_image"])
    sdd_visualize.visualize_training_instance(ax5, instance_dict=instance_dict)
    plot_simulation_step_5(ax5, p_occls, p_disoccls, ego_point, p2_triangles, p1s, p2s)

    # VI. occluder, ego_point visibility
    fig6, ax6 = plt.subplots()
    sdd_visualize.draw_map_numpy(draw_ax=ax6, scene_image=instance_dict["scene_image"])
    sdd_visualize.visualize_training_instance(ax6, instance_dict=instance_dict)
    plot_simulation_step_6(ax6, p_occls, p_disoccls, ego_point, p1s, p2s, occluded_regions)

    # Everything in one picture
    gs_kw = dict(wspace=0.1, hspace=0.1)
    fig, axs = plt.subplots(nrows=2, ncols=3, gridspec_kw=gs_kw)
    [sdd_visualize.draw_map_numpy(draw_ax=ax, scene_image=instance_dict["scene_image"]) for ax in axs.flatten()]
    [sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict, lgnd=False)
     for ax in axs.reshape(-1)[:-1]]
    sdd_visualize.visualize_training_instance(axs[1, 2], instance_dict=instance_dict)
    plot_simulation_step_1(axs[0, 0], agent_visipoly_buffers, no_occluder_buffers, no_ego_buffers, frame_box)
    plot_simulation_step_2(axs[0, 1], agent_visipoly_buffers, no_ego_buffers, frame_box, no_ego_wedges,
                           targets_fullobs_regions, p_occls, p_disoccls)
    plot_simulation_step_3(axs[0, 2], yes_ego_triangles, p_occls, p_disoccls, ego_point, no_occluder_buffers,
                           ego_buffer, p1_area)
    plot_simulation_step_4(axs[1, 0], p_occls, p_disoccls, ego_point, p1_triangles, p1s, p1_visipolys, p2_area)
    plot_simulation_step_5(axs[1, 1], p_occls, p_disoccls, ego_point, p2_triangles, p1s, p2s)
    plot_simulation_step_6(axs[1, 2], p_occls, p_disoccls, ego_point, p1s, p2s, occluded_regions)

    plt.show()


def visualize_random_simulation_samples(
        dataset: StanfordDroneDataset,
        sim_config: dict,
        nrows: int = 2,
        ncols: int = 2
) -> None:
    import logging

    # logging to standard out
    logger = logging.getLogger(__name__)
    c_handler = logging.StreamHandler()
    logger.addHandler(c_handler)

    gs_kw = dict(wspace=0.1, hspace=0.1)
    fig, axs = plt.subplots(nrows, ncols, gridspec_kw=gs_kw)

    n_samples = nrows * ncols
    idx_samples = np.sort(np.random.randint(0, len(dataset), n_samples))

    for i in range(n_samples):
        row, col = i // ncols, i % ncols

        instance_dict = dataset.__getitem__(idx_samples[i])

        ax_i = axs[row, col]

        sdd_visualize.draw_map_numpy(draw_ax=ax_i, scene_image=instance_dict["scene_image"])
        sdd_visualize.visualize_training_instance(draw_ax=ax_i, instance_dict=instance_dict)

        img = instance_dict["scene_image"]
        agents = instance_dict["agents"]
        past_window = instance_dict["past_window"]
        future_window = instance_dict["future_window"]

        try:
            simulation_dict = simulate_occlusions(
                config=sim_config,
                image_res=tuple(img.shape[:2]),
                agents=agents,
                past_window=past_window,
                future_window=future_window
            )

            occlusion_target_coords = simulation_dict["occlusion_target_coords"]
            ego_point = simulation_dict["ego_point"]
            occluders = simulation_dict["occluders"]
            occluded_regions = simulation_dict["occluded_regions"]

            # visualization part
            p_occls = [coords[0] for coords in occlusion_target_coords]
            p_disoccls = [coords[1] for coords in occlusion_target_coords]
            p1s = [occluder[0] for occluder in occluders]
            p2s = [occluder[1] for occluder in occluders]

            plot_simulation_step_6(
                ax=ax_i,
                p_occls=p_occls,
                p_disoccls=p_disoccls,
                ego_point=ego_point,
                p1s=p1s,
                p2s=p2s,
                occluded_regions=occluded_regions
            )

        except Exception as ex:
            logger.exception("\n\nSimulation Failed:\n")
            ax_i.text(
                sum(ax_i.get_xlim())/2, sum(ax_i.get_ylim())/2, "FAILED",
                fontsize=20, c="red", horizontalalignment="center", verticalalignment="center"
            )

    plt.show()


def show_simulation():
    import src.data.sdd_extract as sdd_extract
    import src.visualization.sdd_visualize as sdd_visualize

    config = sdd_extract.get_config("config")
    dataset = StanfordDroneDataset(config_dict=config)

    # showing the simulation process of some desired instance
    instance_idx = 7592     # coupa video0 60
    # instance_idx = 36371    # nexus video7 3024
    # instance_idx = np.random.randint(len(dataset))
    print(f"dataset.__getitem__({instance_idx})")
    instance_dict = dataset.__getitem__(instance_idx)

    fig, ax = plt.subplots()
    sdd_visualize.draw_map_numpy(draw_ax=ax, scene_image=instance_dict["scene_image"])
    sdd_visualize.visualize_training_instance(draw_ax=ax, instance_dict=instance_dict)

    # time_polygon_generation(instance_dict=instance_dict, n_iterations=100000)
    sim_params = config["occlusion_simulator"]
    img = instance_dict["scene_image"]
    agents = instance_dict["agents"]
    past_window = instance_dict["past_window"]
    future_window = instance_dict["future_window"]

    simulation_outputs = simulate_occlusions(
        config=sim_params,
        image_res=tuple(img.shape[:2]),
        agents=agents,
        past_window=past_window,
        future_window=future_window
    )
    visualize_occlusion_simulation(instance_dict, simulation_outputs)

    # Showing the simulation outputs of some random instances
    visualize_random_simulation_samples(dataset, config["occlusion_simulator"], 2, 2)


if __name__ == '__main__':
    show_simulation()
