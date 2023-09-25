import numpy as np
import src.data.config as conf
import src.data.sdd_dataloader as sdd_dataloader


"""
This script provides summary statistics about each scene/video.
"""


if __name__ == '__main__':
    config = conf.get_config("config")
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    print(f"{dataset.lookuptable=}")

    lookuptable = dataset.lookuptable

    print(f"{lookuptable.index.get_level_values('scene').unique()=}")
    for scene in lookuptable.index.get_level_values('scene').unique():
        scene_df = lookuptable.loc[scene]
        # print(f"{len(scene_df)=}")
        # print(f"{scene_df=}")
        for video in scene_df.index.get_level_values('video').unique():
            video_df = scene_df.loc[video]
            instances_n_agents = []
            for index, instance in video_df.iterrows():
                # print(f"{instance['targets']=}")
                instances_n_agents.append(len(instance["targets"]))
            print(f"{scene, video, len(video_df), np.mean(instances_n_agents), np.std(instances_n_agents)=}")
            # print(f"{len(video_df)=}")
            # print(f"{video_df=}")

