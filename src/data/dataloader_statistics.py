import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import src.data.config as conf
import src.data.sdd_dataloader as sdd_dataloader
import src.visualization.sdd_visualize as sdd_visualize

def get_summary_n_agents_per_video() -> None:
    config = conf.get_config("config")
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    print(f"{dataset.lookuptable=}")

    lookuptable = dataset.lookuptable

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


def get_greatest_distance_m():
    config = conf.get_config("config")
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    distance_df = pd.DataFrame(
        columns=['idx', 'scene', 'video', 'timestep', 'n_agents', 'max_dist_px', 'mean_dist_px']
    )

    stop_at = -10
    for instance_idx in tqdm(range(len(dataset))):
        # print(f"{instance_idx=}")
        # print(f"{dataset.__getitem__(instance_idx)=}")
        instance_dict = dataset.__getitem__(instance_idx)

        agent_positions = np.empty([3 * len(instance_dict['agents']), 2])
        for idx, agent in enumerate(instance_dict['agents']):
            p_obs = agent.position_at_timestep(instance_dict['full_window'][0])
            p_zero = agent.position_at_timestep(instance_dict['past_window'][-1])
            p_pred = agent.position_at_timestep(instance_dict['full_window'][-1])

            agent_positions[3*idx, :] = p_obs
            agent_positions[3*idx+1, :] = p_zero
            agent_positions[3*idx+2, :] = p_pred

        # print(f"{agent_positions, len(agent_positions)=}")
        diff_matrix = np.abs(agent_positions[:, np.newaxis] - agent_positions)
        # print(f"{diff_matrix, diff_matrix.shape=}")

        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        # print(f"{dist_matrix, dist_matrix.shape=}")
        dist_matrix = np.triu(dist_matrix)
        # print(f"{dist_matrix, dist_matrix.shape=}")

        dists = dist_matrix[np.nonzero(dist_matrix)]
        # print(f"{dists=}")

        distance_df.loc[len(distance_df)] = [
            instance_idx, instance_dict['scene'], instance_dict['video'], instance_dict['timestep'],
            len(instance_dict['agents']), np.max(dists), np.mean(dists)
        ]

        if instance_idx == stop_at:
            print(f"broke because: {instance_idx=}")
            break

    distance_df['px_ref'] = np.NAN
    distance_df['m_ref'] = np.NAN

    for index, row in distance_df.iterrows():
        # print(f"{index=}")
        # print(f"{row['scene'], row['video']=}")
        # print(f"{conf.PX_PER_M.loc[row['scene'], row['video']]['px']=}")
        distance_df.loc[index, 'px_ref'] = conf.PX_PER_M.loc[row['scene'], row['video']]['px']
        distance_df.loc[index, 'm_ref'] = conf.PX_PER_M.loc[row['scene'], row['video']]['m']

    distance_df['max_dist_m'] = distance_df['max_dist_px'] * distance_df['m_ref'] / distance_df['px_ref']
    distance_df['mean_dist_m'] = distance_df['mean_dist_px'] * distance_df['m_ref'] / distance_df['px_ref']
    # print(f"{distance_df=}")
    row_max_dist = distance_df['max_dist_m'].idxmax()
    print(f"{row_max_dist=}")

    print(f"Maximum inter-agent distance in dataset: {distance_df.iloc[row_max_dist]['max_dist_m']} [m]")
    # idx_ = distance_df.iloc[row_max_dist]['idx']
    # print(f"{idx_=}")
    # instance = dataset.__getitem__(idx_)
    # print(instance)
    # fig, ax = plt.subplots()
    # sdd_visualize.visualize_training_instance(draw_ax=ax, instance_dict=instance)
    # plt.show()


if __name__ == '__main__':
    # get_summary_n_agents_per_video()
    get_greatest_distance_m()
