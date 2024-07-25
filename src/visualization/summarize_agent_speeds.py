import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import src.data.config as conf
from src.data.sdd_agent import StanfordDroneAgent
from src.data import sdd_data_processing


if __name__ == '__main__':

    # TODO: use the proper coordinate conversion framework.

    from tqdm import tqdm
    config_dict = conf.get_config(conf.SDD_CONFIG)
    data_path = config_dict["path"]
    print(f"Extracting data from:\n{data_path}\n")

    # individual agents' walking speeds in px/s
    avg_speeds = []

    for scene_name in tqdm(os.scandir(os.path.join(data_path, "annotations"))):
        for video_name in os.scandir(os.path.realpath(scene_name)):

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

    plt.hist((np.array(avg_speeds) / 2.5), bins=100)
    print(f"{np.mean(avg_speeds)/2.5=} [px/s]")
    print(f"{np.median(avg_speeds)/2.5=} [px/s]")

    avg_walking_speed = 1.42  # [m/s]
    avg_px_per_s = 5.3  # [px/s], reading from histogram

    px_per_m = avg_px_per_s / avg_walking_speed

    print(f"PIXELS PER METER: {px_per_m} [px/m]")  # this value seems inaccurate

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

    geographic_measurements['px/m'] = geographic_measurements['px'] / geographic_measurements['m']

    print(f"{geographic_measurements=}")
    print(f"{np.mean(geographic_measurements['px/m'])=}")
    plt.show()
