import os

import matplotlib.axes
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms

import sdd_extract
import sdd_data_processing


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
    # len(annot_df["frame"].unique())
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

        sdd_data_processing.get_keep_mask_from(agent_df)

        draw_single_trajectory_onto_image(draw_ax=draw_ax, agent_df=agent_df, c=c)


def visualize_training_instance(draw_ax: matplotlib.axes.Axes, **kwargs):
    """
    This function draws the trajectory segments of agents contained within one single training instance extracted from
    the dataloader.

    :param draw_ax: the matplotlib axes object onto which we are drawing
    :return:
    """
    # TODO: maybe add a check to draw partially observed trajectories in gray

    # first draw the reference image onto the axis
    if "image_tensor" in kwargs:
        draw_ax.imshow(kwargs["image_tensor"].permute(1, 2, 0))

    # draw the past trajectories
    if "pasts" in kwargs:
        color_iter = iter(plt.cm.rainbow(np.linspace(0, 1, len(kwargs["pasts"]))))
        # print(kwargs["pasts"])
        for past in kwargs["pasts"]:
            c = next(color_iter).reshape(1, -1)
            draw_ax.plot(past[:, 0], past[:, 1], c=c)
            draw_ax.scatter(past[:, 0], past[:, 1], c=c, marker="+")

    # draw the future trajectories
    if "futures" in kwargs:
        color_iter = iter(plt.cm.rainbow(np.linspace(0, 1, len(kwargs["futures"]))))
        # print(kwargs["futures"])
        for future in kwargs["futures"]:
            c = next(color_iter).reshape(1, -1)
            draw_ax.plot(future[:, 0], future[:, 1], c=c, alpha=0.7)
            draw_ax.scatter(future[:, 0], future[:, 1], c=c, marker="x", alpha=0.7)

def visualize_full_trajectories_on_all_scenes():
    config_dict = sdd_extract.get_config()
    data_path = config_dict["dataset"]["path"]
    print(f"Extracting data from:\n{data_path}\n")

    show = True
    save = False
    save_path = config_dict["results"]["fig_path"]
    assert os.path.exists(save_path)

    for scene_name in os.scandir(os.path.join(data_path, "annotations")):
        for video_name in os.scandir(os.path.realpath(scene_name)):
            # print(os.path.realpath(video_name))
            annot_file_path = os.path.join(os.path.realpath(video_name), "annotations.txt")
            ref_image_path = os.path.join(os.path.realpath(video_name), "reference.jpg")

            annot_file_df = sdd_extract.pd_df_from(annotation_filepath=annot_file_path)

            # n_timesteps = count_timesteps(annot_file_df)
            # duration = timesteps_to_seconds(n_timesteps)
            # mins, sec = duration // 60, duration % 60
            # print(f"{scene_name.name}, {video_name.name}: {n_timesteps} timesteps ({duration}s: {mins}m, {sec}s)")

            annot_file_df = sdd_data_processing.bool_columns_in(annot_file_df)
            annot_file_df = sdd_data_processing.completely_lost_trajs_removed_from(annot_file_df)
            annot_file_df = sdd_data_processing.xy_columns_in(annot_file_df)
            annot_file_df = sdd_data_processing.keep_masks_in(annot_file_df)
            annot_file_df = annot_file_df[annot_file_df["keep"]]
            annot_file_df = sdd_data_processing.subsample_timesteps_from(annot_file_df)

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


def main():
    visualize_full_trajectories_on_all_scenes()


if __name__ == '__main__':
    main()
