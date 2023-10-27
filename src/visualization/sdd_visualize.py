import os
import matplotlib.axes
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch
import skgeom as sg

from typing import List

import src.data.config as conf
import src.data.sdd_data_processing as sdd_data_processing
import src.occlusion_simulation.polygon_generation as poly_gen
import src.occlusion_simulation.visibility as visibility
from src.visualization.plot_utils import plot_sg_polygon
from src.data.sdd_agent import StanfordDroneAgent

SDD_CLASS_SYMBOLS = {'Pedestrian': 'P',     # pedestrian
                     'Biker': 'C',          # cyclist
                     'Skater': 'S',         # skater
                     'Car': 'A',            # automobile
                     'Bus': 'B',            # bus
                     'Cart': 'G'}           # golf cart


def count_timesteps(annot_df: pd.DataFrame):
    """
    shows the number of timesteps recorded for a given scene annotation dataframe
    :param annot_df: the annotation dataframe
    :return: count, the total number of timesteps
    """
    frames = annot_df["frame"].unique()
    n_timesteps = max(frames) - min(frames) + 1
    return n_timesteps


def timesteps_to_seconds(n, fps=30):
    return n/fps


def draw_single_trajectory_onto_image(draw_ax: matplotlib.axes.Axes, agent_df: pd.DataFrame, c):
    agent_class = agent_df["label"].iloc[0]
    frames = agent_df.loc[:, "frame"].values
    manual_annot_idx = (1 - agent_df["gen."].values).astype(bool)
    lost_idx = agent_df["lost"].values.astype(bool)
    occl_idx = agent_df["occl."].values.astype(bool)

    # check that all frames are in ascending order
    assert np.all(frames[:-1] <= frames[1:])

    x = agent_df["x"].values
    y = agent_df["y"].values
    draw_ax.plot(x, y, alpha=0.5, c=c)

    manual_annot_x = x[manual_annot_idx]
    manual_annot_y = y[manual_annot_idx]
    draw_ax.scatter(manual_annot_x, manual_annot_y, s=20, marker='x', alpha=0.8, c=c)

    lost_x = x[lost_idx]
    lost_y = y[lost_idx]
    draw_ax.scatter(lost_x, lost_y, s=20, marker='$?$', alpha=0.8, c=c)

    occl_x = x[occl_idx]
    occl_y = y[occl_idx]
    draw_ax.scatter(occl_x, occl_y, s=20, marker='*', alpha=0.8, c=c)

    last_x = x[frames.argmax()]
    last_y = y[frames.argmax()]
    draw_ax.scatter(last_x, last_y, s=40, marker=f"${SDD_CLASS_SYMBOLS[agent_class]}$", alpha=0.8, c=c)


def draw_all_trajectories_onto_image(draw_ax: matplotlib.axes.Axes, traj_df: pd.DataFrame):

    agents = traj_df["Id"].unique()
    color_iter = iter(plt.cm.rainbow(np.linspace(0, 1, len(agents))))

    for agent in agents:
        agent_df = traj_df[traj_df["Id"] == agent]
        c = next(color_iter).reshape(1, -1)
        draw_single_trajectory_onto_image(draw_ax=draw_ax, agent_df=agent_df, c=c)


def draw_map_torch(draw_ax: matplotlib.axes.Axes, image_tensor: torch.Tensor) -> None:
    # image_tensor.shape = [C, H, W]
    # set the x and y axes
    draw_ax.set_xlim(0., image_tensor.shape[2])
    draw_ax.set_ylim(image_tensor.shape[1], 0.)

    # draw the reference image
    draw_ax.imshow(image_tensor.permute(1, 2, 0))


def draw_map_numpy(draw_ax: matplotlib.axes.Axes, scene_image: np.array) -> None:
    # scene_image.shape = [H, W, C]
    # set the x and y axes
    draw_ax.set_xlim(0., scene_image.shape[1])
    draw_ax.set_ylim(scene_image.shape[0], 0.)

    # draw the image
    draw_ax.imshow(scene_image)


def draw_occlusion_map(
        draw_ax: matplotlib.axes.Axes,
        scene_boundary: sg.Polygon,
        ego_visipoly: sg.Polygon
) -> None:
    occluded_regions = sg.PolygonSet(scene_boundary).difference(ego_visipoly)
    [plot_sg_polygon(ax=draw_ax, poly=poly, edgecolor="red", facecolor="red", alpha=0.2)
     for poly in occluded_regions.polygons]


def draw_occlusion_states(
        draw_ax: matplotlib.axes.Axes,
        agent_trajectories: np.array,
        occlusion_masks: np.array
) -> None:
    # agent_trajectories.shape = [N, T, 2], dtype = float
    # occlusion_masks.shape = [N, T], dtype = bool
    occluded = agent_trajectories[~occlusion_masks]     # [Occluded, 2]
    draw_ax.scatter(occluded[:, 0], occluded[:, 1],
                    s=50, marker="x", color="black", alpha=0.9)


def draw_agent_trajectories(
        draw_ax: matplotlib.axes.Axes,
        agents: List[StanfordDroneAgent],
        past_window: np.array, future_window: np.array
) -> None:
    color_iter = iter(plt.cm.rainbow(np.linspace(0, 1, len(agents))))
    for agent in agents:
        c = next(color_iter).reshape(1, -1)
        past = agent.get_traj_section(time_window=past_window)
        future = agent.get_traj_section(
            time_window=np.array([past_window[-1]] + list(future_window))
        )

        draw_ax.plot(past[:, 0], past[:, 1], color=c)
        draw_ax.scatter(past[:-1, 0], past[:-1, 1], s=20, marker="x", color=c)
        draw_ax.scatter(past[-1, 0], past[-1, 1],
                        s=40, marker="x", color=c, label=f"${SDD_CLASS_SYMBOLS[agent.label]}^{{{agent.id}}}$")
        draw_ax.plot(future[:, 0], future[:, 1], color=c, linestyle="dashed", alpha=0.8)
        draw_ax.scatter(future[1:, 0], future[1:, 1], s=20, marker="x", color=c, alpha=0.8)


def visualize_training_instance(draw_ax: matplotlib.axes.Axes, instance_dict: dict, lgnd: bool = True) -> None:
    """
    This function draws the trajectory segments of agents contained within one single training instance extracted from
    the dataloader.

    :param draw_ax: the matplotlib axes object onto which we are drawing
    :param instance_dict: dictionary containing the following:
        - 'agents': a list of StanfordDroneAgent objects, instantiated through the class defined in sdd_dataloader.py
        - 'past_window': a numpy array of type 'int', indicating the timesteps corresponding to the observation window
        - 'future_window': a numpy array of type 'int', indicating the timesteps corresponding to the prediction horizon
        - 'scene_image': a numpy array containing the reference image data
    :param lgnd: whether to display the legend or not
    """
    assert all(key in instance_dict.keys() for key in ["agents", "past_window", "future_window", "full_window"])

    if "scene_image" in instance_dict.keys():
        draw_map_numpy(draw_ax=draw_ax, scene_image=instance_dict["scene_image"])

    if not np.all(np.isnan(instance_dict.get('ego_point'))):
        draw_ax.scatter(instance_dict["ego_point"][0], instance_dict["ego_point"][1],
                        s=20, marker="D", color="yellow", alpha=0.9, label="Ego")

    if not np.all(np.isnan(instance_dict.get('occluders'))):
        for occluder in instance_dict["occluders"]:
            draw_ax.plot([occluder[0][0], occluder[1][0]], [occluder[0][1], occluder[1][1]],
                         color="black")

    if all(not np.all(np.isnan(instance_dict.get(key))) for key in ["scene_image", "ego_point", "occluders"]):
        # compute visibility polygon
        scene_boundary = poly_gen.default_rectangle(
            (float(instance_dict["scene_image"].shape[0]),
             float(instance_dict["scene_image"].shape[1]))
        )

        ego_visipoly = visibility.compute_visibility_polygon(
            ego_point=instance_dict["ego_point"],
            occluders=instance_dict["occluders"],
            boundary=scene_boundary
        )

        draw_occlusion_map(draw_ax=draw_ax, scene_boundary=scene_boundary, ego_visipoly=ego_visipoly)

        # compute occlusion masks
        full_window_occlusion_masks = visibility.agent_occlusion_masks(
            agents=instance_dict["agents"],
            time_window=instance_dict["full_window"],
            ego_visipoly=ego_visipoly
        )
        draw_occlusion_states(
            draw_ax=draw_ax,
            agent_trajectories=np.stack(
                [agent.get_traj_section(instance_dict["full_window"])
                 for agent in instance_dict["agents"]]
            ),
            occlusion_masks=full_window_occlusion_masks
        )

    draw_agent_trajectories(
        draw_ax=draw_ax,
        agents=instance_dict["agents"],
        past_window=instance_dict["past_window"],
        future_window=instance_dict["future_window"]
    )

    if lgnd:
        draw_ax.legend(fancybox=True, framealpha=0.2, fontsize=10)


def visualize_full_trajectories_on_all_scenes():
    config_dict = conf.get_config("config")
    data_path = config_dict["dataset"]["path"]
    print(f"Extracting data from:\n{data_path}\n")

    show = True
    save = False
    save_path = os.path.abspath(os.path.join(conf.REPO_ROOT, config_dict["results"]["fig_path"]))
    assert os.path.exists(save_path)

    for scene_name in os.scandir(os.path.join(data_path, "annotations")):
        for video_name in os.scandir(os.path.realpath(scene_name)):
            print(os.path.realpath(video_name))
            annot_file_path = os.path.join(os.path.realpath(video_name), "annotations.txt")
            ref_image_path = os.path.join(os.path.realpath(video_name), "reference.jpg")

            annot_file_df = sdd_data_processing.pd_df_from(annotation_filepath=annot_file_path)

            # n_timesteps = count_timesteps(annot_file_df)
            # duration = timesteps_to_seconds(n_timesteps)
            # mins, sec = duration // 60, duration % 60
            # print(f"{scene_name.name}, {video_name.name}: {n_timesteps} timesteps ({duration}s: {mins}m, {sec}s)")

            if config_dict["hyperparameters"]["other_agents"] == "OUT":
                annot_file_df = annot_file_df[
                    annot_file_df["label"].isin(config_dict["hyperparameters"]["agent_types"])
                ]
            annot_file_df = sdd_data_processing.perform_preprocessing_pipeline(
                annot_df=annot_file_df,
                target_fps=config_dict["hyperparameters"]["fps"],
                orig_fps=config_dict["dataset"]["fps"]
            )

            # with pd.option_context('display.max_rows', None,
            #                        'display.max_columns', None,
            #                        'display.precision', 3,):
            #     print(annot_file_df[["Id", "lost", "keep"]])

            # # we only care about pedestrians:
            # is_pedestrian = annot_file_df["label"] == "Pedestrian"
            # annot_file_df = annot_file_df[is_pedestrian]

            # # we highlight manually annotated points
            # not_is_gen = annot_file_df["gen."] == 0
            # annot_file_df = annot_file_df[not_is_gen]

            fig, ax = plt.subplots()
            fig.canvas.manager.set_window_title(f"{scene_name.name}: {video_name.name}")

            # drawing the example image onto the figure
            ref_image = cv2.imread(ref_image_path)
            ax.imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))

            # drawing trajectories onto the figure
            draw_all_trajectories_onto_image(draw_ax=ax, traj_df=annot_file_df)

            if save:
                plt.savefig(os.path.join(save_path, f"{scene_name.name}-{video_name.name}"), bbox_inches="tight")
                plt.close()

            if show:
                plt.show(block=True)
                # plt.pause(5)
                plt.close()


def avg_walking_speed_px():
    from tqdm import tqdm
    config_dict = conf.get_config("config")
    data_path = config_dict["dataset"]["path"]
    print(f"Extracting data from:\n{data_path}\n")

    # individual agents' walking speeds in px/s
    avg_speeds = []

    for scene_name in tqdm(os.scandir(os.path.join(data_path, "annotations"))):
        for video_name in os.scandir(os.path.realpath(scene_name)):
            # print(f"{os.path.realpath(video_name)=}")

            annot_file_path = os.path.join(os.path.realpath(video_name), "annotations.txt")
            annot_file_df = sdd_data_processing.pd_df_from(annotation_filepath=annot_file_path)

            annot_file_df = annot_file_df[annot_file_df['label'] == 'Pedestrian']
            annot_file_df = sdd_data_processing.perform_preprocessing_pipeline(
                annot_df=annot_file_df,
                target_fps=2.5, orig_fps=30
            )

            agents = [StanfordDroneAgent(annot_file_df[annot_file_df["Id"] == agent_id])
                      for agent_id in annot_file_df["Id"].unique()]

            for agent in agents:
                traj = agent.fulltraj
                if len(traj) < 2:
                    # print(f"CONTINUE")
                    continue
                velocity_profile = traj[1:] - traj[:-1]
                speed_profile = np.linalg.norm(velocity_profile, axis=1)
                avg_speed = np.mean(speed_profile)
                avg_speeds.append(avg_speed)

    plt.hist((np.array(avg_speeds)/2.5), bins=100)
    print(f"{np.mean(avg_speeds)/2.5=} [px/s]")
    print(f"{np.median(avg_speeds)/2.5=} [px/s]")

    avg_walking_speed = 1.42    # [m/s]
    avg_px_per_s = 5.3          # [px/s], reading from histogram

    px_per_m = avg_px_per_s / avg_walking_speed

    print(f"PIXELS PER METER: {px_per_m} [px/m]")       # this value seems inaccurate

    # instead, we prefer to focus on using landmarks in the videos to measure distances in pixels, which we can
    # compare to the real, physical distances that we can measure in the real world
    # (physical measurements are obtained thanks to satellite collected geographic data)
    measured_data = [
        ['bookstore', 668.4, 25.3],
        ['coupa', 814.3, 22.7],
        ['deathCircle', 422.1, 17.47],
        ['gates', 544.3, 20.88],
        ['hyang', 933.0, 31.73],
        ['little', 677.4, 19.81],
        ['nexus', 783.0, 34.03],
        ['quad', 820.1, 35.94],
    ]
    geographic_measurements = pd.DataFrame(measured_data, columns=['scene', 'px', 'm'])

    geographic_measurements['px/m'] = geographic_measurements['px']/geographic_measurements['m']

    print(f"{geographic_measurements=}")
    print(f"{np.mean(geographic_measurements['px/m'])=}")
    plt.show()


def get_video_resolutions():
    config = conf.get_config("config")
    data_dir = os.path.join(config["dataset"]["path"], "annotations")
    print(data_dir)
    assert os.path.exists(data_dir)

    image_summaries_df = pd.DataFrame(
        columns=[
            'scene', 'video',
            'width [px]', 'height [px]', 'diagonal [px]',
            'width [m]', 'height [m]', 'diagonal [m]',
            'aspect ratio'
        ]
    )

    counter = 0
    for dir in os.scandir(data_dir):
        for video in os.scandir(dir):
            file = os.path.join(video, "reference.jpg")
            assert os.path.exists(file)
            counter += 1
            with open(file, "rb") as img_file:
                img_file.seek(163)

                a = img_file.read(2)
                height = (a[0] << 8) + a[1]
                a = img_file.read(2)
                width = (a[0] << 8) + a[1]

                # print(f"{os.path.basename(file)}: {width} x {height}")
                m_by_px = conf.COORD_CONV.loc[dir.name, video.name]['m/px']

                width_m = width * m_by_px
                height_m = height * m_by_px

                image_summaries_df.loc[len(image_summaries_df)] = [
                    dir.name, video.name,
                    width, height, np.sqrt(width**2 + height**2),
                    width_m, height_m, np.sqrt(width_m**2 + height_m**2),
                    np.min([width, height]) / np.max([width, height])
                ]
    image_summaries_df.set_index(keys=['scene', 'video'], inplace=True)
    image_summaries_df.sort_index(inplace=True)
    print("Image resolution summary:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(image_summaries_df)

    print()
    print(f"min width [px]: {min(image_summaries_df['width [px]'])}")
    print(f"max width [px]: {max(image_summaries_df['width [px]'])}")
    print(f"min height [px]: {min(image_summaries_df['height [px]'])}")
    print(f"max height [px]: {max(image_summaries_df['height [px]'])}")


def copy_reference_images_to_figures_dir():
    import shutil
    config_dict = conf.get_config("config")
    data_path = config_dict["dataset"]["path"]
    output_path = os.path.join(conf.REPO_ROOT, config_dict["results"]["fig_path"], "reference_images")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for scene in os.scandir(os.path.join(data_path, "annotations")):
        for video in os.scandir(os.path.realpath(scene)):

            img_file = os.path.join(os.path.realpath(video), "reference.jpg")
            assert os.path.exists(img_file)
            out_file = os.path.join(output_path, f"{scene.name}_{video.name}.jpg")
            shutil.copy(img_file, out_file)


def main():
    # visualize_full_trajectories_on_all_scenes()
    get_video_resolutions()
    # avg_walking_speed_px()
    # copy_reference_images_to_figures_dir()


if __name__ == '__main__':
    main()
