import argparse
import numpy as np
import pandas as pd

import src.data.config as conf
import src.data.sdd_dataloader as sdd_dataloader


def summarize_agent_counts_per_video(
        dataset_cfg: str
) -> None:
    config = conf.get_config(dataset_cfg)
    dataset = sdd_dataloader.StanfordDroneDataset(config=config)

    lookuptable = dataset.lookuptable

    out_df = pd.DataFrame(
        columns=['scene', 'video', '# instances', 'avg. # agents / instance', 'std. # agents / instance']
    )

    for scene in lookuptable.index.get_level_values('scene').unique():
        scene_df = lookuptable.loc[scene]
        for video in scene_df.index.get_level_values('video').unique():
            video_df = scene_df.loc[video]
            instances_n_agents = []
            for index, instance in video_df.iterrows():
                instances_n_agents.append(len(instance["targets"]))
            # print(f"{scene, video, len(video_df), np.mean(instances_n_agents), np.std(instances_n_agents)=}")

            out_df.loc[len(out_df)] = {
                'scene': scene,
                'video': video,
                '# instances': len(video_df),
                'avg. # agents / instance': np.mean(instances_n_agents),
                'std. # agents / instance': np.std(instances_n_agents)
            }

    print(out_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-cfg', type=str, required=True,
                        help='name of the .yaml config file to use for the parameters of the base SDD dataset.')
    args = parser.parse_args()

    summarize_agent_counts_per_video(args.dataset_cfg)
