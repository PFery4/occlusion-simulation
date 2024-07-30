import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
from tqdm import tqdm

import src.data.config as conf
import src.data.sdd_dataloader as sdd_dataloader


def get_greatest_distance_m(
        dataset_cfg: str,
        save: bool,
        show: bool
):

    config = conf.get_config(dataset_cfg)
    dataset = sdd_dataloader.StanfordDroneDataset(config=config)

    distance_df = pd.DataFrame(
        columns=[
            'idx', 'scene', 'video', 'timestep', 'n_agents',
            'max_dist_inter_agent_px', 'mean_dist_inter_agent_px', 'max_dist_mean_zero_px', 'mean_dist_mean_zero_px'
        ]
    )

    for instance_idx in tqdm(range(len(dataset))):
        instance_dict = dataset.__getitem__(instance_idx)

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

        diff_mean_zero = np.abs(positions - mean_zero)
        diff_matrix = np.abs(positions[:, np.newaxis] - positions)

        dist_mean_zero = np.linalg.norm(diff_mean_zero, axis=1)
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        dist_matrix = np.triu(dist_matrix)
        dists = dist_matrix[np.nonzero(dist_matrix)]
        if not np.any(dists):
            dists = np.array([0.0])

        distance_df.loc[len(distance_df)] = [
            instance_idx, instance_dict['scene'], instance_dict['video'], instance_dict['timestep'],
            len(instance_dict['agents']), np.max(dists), np.mean(dists), np.max(dist_mean_zero), np.mean(dist_mean_zero)
        ]

    distance_df['px_ref'] = np.NAN
    distance_df['m_ref'] = np.NAN

    for index, row in tqdm(distance_df.iterrows()):
        distance_df.loc[index, 'm/px'] = conf.COORD_CONV.loc[row['scene'], row['video']]['m/px']

    distance_df['max_dist_inter_agent_m'] = distance_df['max_dist_inter_agent_px'] * distance_df['m/px']
    distance_df['mean_dist_inter_agent_m'] = distance_df['mean_dist_inter_agent_px'] * distance_df['m/px']
    distance_df['max_dist_mean_zero_m'] = distance_df['max_dist_mean_zero_px'] * distance_df['m/px']
    distance_df['mean_dist_mean_zero_m'] = distance_df['mean_dist_mean_zero_px'] * distance_df['m/px']

    row_max_dist_inter_agent = distance_df['max_dist_inter_agent_m'].idxmax()
    row_max_dist_mean_zero = distance_df['max_dist_mean_zero_m'].idxmax()

    summarystr = f"Maximum inter-agent distance in dataset:\n"\
                 f"dataset index: {distance_df.iloc[row_max_dist_inter_agent]['idx']}\n"\
                 f"{distance_df.iloc[row_max_dist_inter_agent]['max_dist_inter_agent_m']} [m]\n\n"\
                 f"Maximum distance to mean position at t=0 in dataset:\n"\
                 f"dataset index: {distance_df.iloc[row_max_dist_mean_zero]['idx']}\n"\
                 f"{distance_df.iloc[row_max_dist_mean_zero]['max_dist_mean_zero_m']} [m]\n"

    print(summarystr)

    fig, ax = plt.subplots(2, 4)

    # Pixels
    scatter_kwargs = {'marker': '.', 'alpha': 0.6, 'linewidths': 0.0}
    ax[0, 0].scatter(distance_df['n_agents'], distance_df['mean_dist_inter_agent_px'], c='r', **scatter_kwargs)
    ax[0, 0].set_title('Mean inter-agent distance')
    ax[0, 0].set_xlabel('Number of agents')
    ax[0, 0].set_ylabel('Mean inter-agent distance [px]')

    ax[0, 1].scatter(distance_df['n_agents'], distance_df['max_dist_inter_agent_px'], c='r', **scatter_kwargs)
    ax[0, 1].set_title('Max inter-agent distance')
    ax[0, 1].set_xlabel('Number of agents')
    ax[0, 1].set_ylabel('Max inter-agent distance [px]')

    ax[1, 0].scatter(distance_df['n_agents'], distance_df['mean_dist_mean_zero_px'], c='r', **scatter_kwargs)
    ax[1, 0].set_title('Mean distance to avg. position at t0')
    ax[1, 0].set_xlabel('Number of agents')
    ax[1, 0].set_ylabel('Mean distance to avg. position at t0 [px]')

    ax[1, 1].scatter(distance_df['n_agents'], distance_df['max_dist_mean_zero_px'], c='r', **scatter_kwargs)
    ax[1, 1].set_title('Max distance to avg. position at t0')
    ax[1, 1].set_xlabel('Number of agents')
    ax[1, 1].set_ylabel('Max distance to avg. position at t0 [px]')

    # Meters
    ax[0, 2].scatter(distance_df['n_agents'], distance_df['mean_dist_inter_agent_m'], c='b', **scatter_kwargs)
    ax[0, 2].set_title('Mean inter-agent distance')
    ax[0, 2].set_xlabel('Number of agents')
    ax[0, 2].set_ylabel('Mean inter-agent distance [m]')

    ax[0, 3].scatter(distance_df['n_agents'], distance_df['max_dist_inter_agent_m'], c='b', **scatter_kwargs)
    ax[0, 3].set_title('Max inter-agent distance')
    ax[0, 3].set_xlabel('Number of agents')
    ax[0, 3].set_ylabel('Max inter-agent distance [m]')

    ax[1, 2].scatter(distance_df['n_agents'], distance_df['mean_dist_mean_zero_m'], c='b', **scatter_kwargs)
    ax[1, 2].set_title('Mean distance to avg. position at t0')
    ax[1, 2].set_xlabel('Number of agents')
    ax[1, 2].set_ylabel('Mean distance to avg. position at t0 [m]')

    ax[1, 3].scatter(distance_df['n_agents'], distance_df['max_dist_mean_zero_m'], c='b', **scatter_kwargs)
    ax[1, 3].set_title('Max distance to avg. position at t0')
    ax[1, 3].set_xlabel('Number of agents')
    ax[1, 3].set_ylabel('Max distance to avg. position at t0 [m]')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.94, top=0.94, wspace=0.28, hspace=0.2)

    if save:
        save_path = os.path.join(conf.REPO_ROOT, 'outputs', 'figures', 'distances')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        assert os.path.exists(save_path)
        print(f"saving figure of distances to:\n{save_path}")
        fig.set_size_inches(20, 12)
        plt.savefig(os.path.join(save_path, 'distances_figure.png'), bbox_inches='tight', dpi=100)
        with open(os.path.join(save_path, 'distances_summary.txt'), 'w') as txt_file:
            print(summarystr, file=txt_file)

    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-cfg',
        type=os.path.abspath, default=os.path.join(conf.REPO_ROOT, 'config', 'dataset_config.yaml'),
        help='name of the .yaml config file to use for the parameters of the base SDD dataset.'
    )
    parser.add_argument(
        '--show', action='store_true', default=False,
        help='whether to display this script\'s output on the screen.'
    )
    parser.add_argument(
        '--save', action='store_true', default=False,
        help='whether to save this script\'s output as a png file.'
    )
    args = parser.parse_args()

    assert args.save or args.show, "Please select at least one option from --save / --show."

    get_greatest_distance_m(
        dataset_cfg=args.dataset_cfg,
        save=args.save,
        show=args.show
    )
