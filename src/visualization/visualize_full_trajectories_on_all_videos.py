import argparse
import cv2
import matplotlib.pyplot as plt
import os

import src.data.config as conf
from src.data import sdd_data_processing
from src.visualization.sdd_visualize import draw_all_trajectories_onto_image


def visualize_full_trajectories_on_all_scenes(preprocessing_cfg: str):

    sdd_config = conf.get_config(conf.SDD_CONFIG)
    config_dict = conf.get_config(preprocessing_cfg)
    data_path = sdd_config["path"]
    print(f"Extracting data from:\n{data_path}\n")

    show = True
    save = False
    save_path = os.path.abspath(os.path.join(conf.REPO_ROOT, 'outputs', 'figures'))
    assert os.path.exists(save_path)

    for scene_name in os.scandir(os.path.join(data_path, "annotations")):
        for video_name in os.scandir(os.path.realpath(scene_name)):
            print(os.path.realpath(video_name))
            annot_file_path = os.path.join(os.path.realpath(video_name), "annotations.txt")
            ref_image_path = os.path.join(os.path.realpath(video_name), "reference.jpg")

            annot_file_df = sdd_data_processing.pd_df_from(annotation_filepath=annot_file_path)

            if config_dict["other_agents"] == "OUT":
                annot_file_df = annot_file_df[
                    annot_file_df["label"].isin(config_dict["agent_types"])
                ]
            annot_file_df = sdd_data_processing.perform_preprocessing_pipeline(
                annot_df=annot_file_df,
                target_fps=config_dict["fps"],
                orig_fps=sdd_config["fps"]
            )

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing-cfg', type=str, required=True,
                        help='name of the .yaml config file to use for the parameters of the preprocessing.')
    args = parser.parse_args()

    visualize_full_trajectories_on_all_scenes(preprocessing_cfg=args.preprocessing_cfg)
