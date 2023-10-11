import os.path

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
    stop_at = -10
    save = True
    show = not save

    config = conf.get_config("config")
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    distance_df = pd.DataFrame(
        columns=[
            'idx', 'scene', 'video', 'timestep', 'n_agents',
            'max_dist_inter_agent_px', 'mean_dist_inter_agent_px', 'max_dist_mean_zero_px', 'mean_dist_mean_zero_px'
        ]
    )

    for instance_idx in tqdm(range(len(dataset))):
        # print(f"{instance_idx=}")
        # print(f"{dataset.__getitem__(instance_idx)=}")
        instance_dict = dataset.__getitem__(instance_idx)

        # agent_positions = np.empty([3 * len(instance_dict['agents']), 2])
        ps_obs = np.empty([len(instance_dict['agents']), 2])
        ps_zero = np.empty([len(instance_dict['agents']), 2])
        ps_pred = np.empty([len(instance_dict['agents']), 2])

        for idx, agent in enumerate(instance_dict['agents']):
            p_obs = agent.position_at_timestep(instance_dict['full_window'][0])
            p_zero = agent.position_at_timestep(instance_dict['past_window'][-1])
            p_pred = agent.position_at_timestep(instance_dict['full_window'][-1])

            ps_obs[idx, :] = p_obs
            ps_zero[idx, :] = p_zero
            ps_pred[idx, :] = p_pred

        mean_zero = np.mean(ps_zero, axis=0)[np.newaxis, :]
        positions = np.concatenate([ps_obs, ps_zero, ps_pred], axis=0)

        # print(f"{ps_obs, ps_obs.shape=}")
        # print(f"{ps_zero, ps_zero.shape=}")
        # print(f"{ps_pred, ps_pred.shape=}")
        # print(f"{mean_zero, mean_zero.shape=}")
        # print(f"{positions, positions.shape=}")

        diff_mean_zero = np.abs(positions - mean_zero)
        diff_matrix = np.abs(positions[:, np.newaxis] - positions)
        # print(f"{diff_mean_zero, diff_mean_zero.shape=}")
        # print(f"{diff_matrix, diff_matrix.shape=}")

        dist_mean_zero = np.linalg.norm(diff_mean_zero, axis=1)
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        dist_matrix = np.triu(dist_matrix)
        dists = dist_matrix[np.nonzero(dist_matrix)]
        # print(f"{dist_mean_zero, dist_mean_zero.shape=}")
        # print(f"{dist_matrix, dist_matrix.shape=}")
        # print(f"{dists=}")

        distance_df.loc[len(distance_df)] = [
            instance_idx, instance_dict['scene'], instance_dict['video'], instance_dict['timestep'],
            len(instance_dict['agents']), np.max(dists), np.mean(dists), np.max(dist_mean_zero), np.mean(dist_mean_zero)
        ]

        if instance_idx == stop_at:
            print(f"broke because: {instance_idx=}")
            break

    distance_df['px_ref'] = np.NAN
    distance_df['m_ref'] = np.NAN

    for index, row in tqdm(distance_df.iterrows()):
        # print(f"{index=}")
        # print(f"{row['scene'], row['video']=}")
        # print(f"{conf.PX_PER_M.loc[row['scene'], row['video']]['px']=}")
        distance_df.loc[index, 'px_ref'] = conf.PX_PER_M.loc[row['scene'], row['video']]['px']
        distance_df.loc[index, 'm_ref'] = conf.PX_PER_M.loc[row['scene'], row['video']]['m']

    distance_df['m/px'] = distance_df['m_ref'] / distance_df['px_ref']

    distance_df['max_dist_inter_agent_m'] = distance_df['max_dist_inter_agent_px'] * distance_df['m/px']
    distance_df['mean_dist_inter_agent_m'] = distance_df['mean_dist_inter_agent_px'] * distance_df['m/px']
    distance_df['max_dist_mean_zero_m'] = distance_df['max_dist_mean_zero_px'] * distance_df['m/px']
    distance_df['mean_dist_mean_zero_m'] = distance_df['mean_dist_mean_zero_px'] * distance_df['m/px']

    # print(f"{distance_df=}")
    row_max_dist_inter_agent = distance_df['max_dist_inter_agent_m'].idxmax()
    row_max_dist_mean_zero = distance_df['max_dist_mean_zero_m'].idxmax()
    # print(f"{row_max_dist_inter_agent=}")
    # print(f"{row_max_dist_mean_zero=}")

    print(f"Maximum inter-agent distance in dataset:\n"
          f"dataset index: {distance_df.iloc[row_max_dist_inter_agent]['idx']}\n"
          f"{distance_df.iloc[row_max_dist_inter_agent]['max_dist_inter_agent_m']} [m]\n")
    print(f"Maximum inter-agent distance in dataset:\n"
          f"dataset index: {distance_df.iloc[row_max_dist_mean_zero]['idx']}\n"
          f"{distance_df.iloc[row_max_dist_mean_zero]['max_dist_mean_zero_m']} [m]\n")

    fig, ax = plt.subplots(2, 4)

    # Pixels
    ax[0, 0].scatter(distance_df['n_agents'], distance_df['mean_dist_inter_agent_px'], marker='x', c='r')
    ax[0, 0].set_title('Mean inter-agent distance')
    ax[0, 0].set_xlabel('Number of agents')
    ax[0, 0].set_ylabel('Mean inter-agent distance [px]')

    ax[0, 1].scatter(distance_df['n_agents'], distance_df['max_dist_inter_agent_px'], marker='x', c='r')
    ax[0, 1].set_title('Max inter-agent distance')
    ax[0, 1].set_xlabel('Number of agents')
    ax[0, 1].set_ylabel('Max inter-agent distance [px]')

    ax[1, 0].scatter(distance_df['n_agents'], distance_df['mean_dist_mean_zero_px'], marker='x', c='r')
    ax[1, 0].set_title('Mean distance to avg. position at t0')
    ax[1, 0].set_xlabel('Number of agents')
    ax[1, 0].set_ylabel('Mean distance to avg. position at t0 [px]')

    ax[1, 1].scatter(distance_df['n_agents'], distance_df['max_dist_mean_zero_px'], marker='x', c='r')
    ax[1, 1].set_title('Max distance to avg. position at t0')
    ax[1, 1].set_xlabel('Number of agents')
    ax[1, 1].set_ylabel('Max distance to avg. position at t0 [px]')

    # Meters
    ax[0, 2].scatter(distance_df['n_agents'], distance_df['mean_dist_inter_agent_m'], marker='x', c='b')
    ax[0, 2].set_title('Mean inter-agent distance')
    ax[0, 2].set_xlabel('Number of agents')
    ax[0, 2].set_ylabel('Mean inter-agent distance [m]')

    ax[0, 3].scatter(distance_df['n_agents'], distance_df['max_dist_inter_agent_m'], marker='x', c='b')
    ax[0, 3].set_title('Max inter-agent distance')
    ax[0, 3].set_xlabel('Number of agents')
    ax[0, 3].set_ylabel('Max inter-agent distance [m]')

    ax[1, 2].scatter(distance_df['n_agents'], distance_df['mean_dist_mean_zero_m'], marker='x', c='b')
    ax[1, 2].set_title('Mean distance to avg. position at t0')
    ax[1, 2].set_xlabel('Number of agents')
    ax[1, 2].set_ylabel('Mean distance to avg. position at t0 [m]')

    ax[1, 3].scatter(distance_df['n_agents'], distance_df['max_dist_mean_zero_m'], marker='x', c='b')
    ax[1, 3].set_title('Max distance to avg. position at t0')
    ax[1, 3].set_xlabel('Number of agents')
    ax[1, 3].set_ylabel('Max distance to avg. position at t0 [m]')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.94, top=0.94, wspace=0.28, hspace=0.2)
    # plt.subplot_tool()

    if save:
        save_path = os.path.join(conf.REPO_ROOT, config['results']['fig_path'])
        assert os.path.exists(save_path)
        print(f"saving figure of distances to:\n{save_path}")
        # print(plt.get_backend())
        # manager = plt.get_current_fig_manager()
        # manager.resize(*manager.window.maxsize())
        fig.set_size_inches(20, 12)
        plt.savefig(os.path.join(save_path, 'distances_figure.png'), bbox_inches='tight', dpi=100)

    if show:
        plt.show()

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
