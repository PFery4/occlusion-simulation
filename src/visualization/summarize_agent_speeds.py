import matplotlib.pyplot as plt
import numpy as np
import os

import src.data.config as conf
from src.data.sdd_agent import StanfordDroneAgent
from src.data import sdd_data_processing


if __name__ == '__main__':

    from tqdm import tqdm
    config_dict = conf.get_config(conf.SDD_CONFIG)
    data_path = config_dict["path"]
    print(f"Extracting data from:\n{data_path}\n")

    coord_conv = conf.COORD_CONV

    fig, ax = plt.subplots(1, 2)

    # individual agents' walking speeds (/s)
    avg_speeds_px = []
    avg_speeds_m = []

    for scene_name in tqdm(os.scandir(os.path.join(data_path, "annotations"))):
        for video_name in os.scandir(os.path.realpath(scene_name)):

            coord_conv_row = coord_conv.loc[
                os.path.basename(scene_name),
                os.path.basename(video_name)
            ]

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
                traj_px = agent.fulltraj
                if len(traj_px) < 2:
                    # print(f"CONTINUE")
                    continue

                velocity_profile_px = traj_px[1:] - traj_px[:-1]
                speed_profile_px = np.linalg.norm(velocity_profile_px, axis=1)
                avg_speeds_px.append(np.mean(speed_profile_px))

                traj_m = traj_px * coord_conv_row['m/px']
                velocity_profile_m = traj_m[1:] - traj_m[:-1]
                speed_profile_m = np.linalg.norm(velocity_profile_m, axis=1)
                avg_speeds_m.append(np.mean(speed_profile_m))

    ax[0].hist((np.array(avg_speeds_m) / 2.5), bins=100, color='b')
    ax[1].hist((np.array(avg_speeds_px) / 2.5), bins=100, color='r')

    plt.show()

    print(f"Mean velocity:  \t{np.mean(avg_speeds_m)/2.5} [m/s]\t\t{np.mean(avg_speeds_px)/2.5} [px/s]")
    print(f"Median velocity:\t{np.median(avg_speeds_m)/2.5} [m/s]\t\t{np.median(avg_speeds_px)/2.5} [px/s]")
