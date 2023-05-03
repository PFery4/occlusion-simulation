import os

import matplotlib.axes
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

import sdd_extract


def draw_all_trajectories_onto_image(draw_ax: matplotlib.axes.Axes, traj_df: pd.DataFrame):

    agents = traj_df["Id"].unique()
    for agent in agents:
        x = traj_df[traj_df["Id"] == agent].loc[:, "x"].values
        y = traj_df[traj_df["Id"] == agent].loc[:, "y"].values
        draw_ax.plot(x, y, alpha=0.5)
        # draw_ax.scatter(x, y, s=5, marker='o')


def main():
    data_path = sdd_extract.get_config()["dataset"]["path"]
    print(f"Extracting data from:\n{data_path}\n")

    for scene_name in os.scandir(os.path.join(data_path, "annotations")):
        # print(os.path.realpath(scene_name))
        for video_name in os.scandir(os.path.realpath(scene_name)):
            annot_file_path = os.path.join(data_path, "annotations", scene_name, video_name, "annotations.txt")
            ref_image_path = os.path.join(data_path, "annotations", scene_name, video_name, "reference.jpg")

            annot_file_df = sdd_extract.pd_df_from(annotation_filepath=annot_file_path)

            sdd_extract.add_xy_columns_to(annot_file_df)

            # # we only care about pedestrians:
            # is_pedestrian = annot_file_df["label"] == "Pedestrian"
            # # we highlight manually annotated points
            # not_is_gen = annot_file_df["gen."] == 0
            # annot_file_df = annot_file_df[is_pedestrian]
            # annot_file_df = annot_file_df[not_is_gen]

            fig, ax = plt.subplots()
            fig.canvas.manager.set_window_title(f"{scene_name.name}: {video_name.name}")
            # fig.suptitle()

            # drawing the example image onto the figure
            ref_image = cv2.imread(ref_image_path)
            ax.imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))

            # drawing trajectories onto the figure
            draw_all_trajectories_onto_image(draw_ax=ax, traj_df=annot_file_df)

        plt.show()


if __name__ == '__main__':
    main()
