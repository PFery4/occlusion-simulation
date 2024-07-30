import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

import src.data.config as conf
from src.data.sdd_dataloader import StanfordDroneDataset
from src.occlusion_simulation.simple_occlusion import simulate_occlusions
import src.visualization.sdd_visualize as sdd_visualize
from src.visualization.simulation_visualize import plot_simulation_step_6


def save_simulation_cases(
        dataset_cfg: str,
        simulator_cfg: str,
        n_examples: int = 100,
        clear_figure_folder: bool = True
):
    dataset_config = conf.get_config(dataset_cfg)
    simulator_config = conf.get_config(simulator_cfg)

    logger = logging.getLogger(__name__)
    c_handler = logging.StreamHandler()
    logger.addHandler(c_handler)

    dataset = StanfordDroneDataset(config=dataset_config)

    # defining path and creating it if it does not exit
    save_path = os.path.join(conf.REPO_ROOT, 'outputs', 'figures', 'simulation_examples')
    os.makedirs(save_path, exist_ok=True)

    # clearing figures
    if clear_figure_folder:
        for item in os.listdir(save_path):
            if item.endswith(".png"):
                print(f"Removing: {item}")
                os.remove(os.path.join(save_path, item))

    # sampling instances
    instance_indices = np.sort(np.random.choice(len(dataset), size=n_examples, replace=False))
    print(f"{instance_indices=}")

    # setting up error counter
    err_count = 0

    for idx in tqdm(instance_indices):
        gs_kw = dict(wspace=0.0, hspace=0.0)
        fig, ax = plt.subplots(gridspec_kw=gs_kw)
        instance_dict = dataset.__getitem__(idx)

        sdd_visualize.visualize_training_instance(draw_ax=ax, instance_dict=instance_dict, lgnd=False)

        img = instance_dict["scene_image"]
        agents = instance_dict["agents"]
        past_window = instance_dict["past_window"]
        future_window = instance_dict["future_window"]

        try:
            simulation_dict = simulate_occlusions(
                config=simulator_config,
                image_res=tuple(img.shape[:2]),
                agents=agents,
                past_window=past_window,
                future_window=future_window
            )

            occlusion_target_coords = simulation_dict["occlusion_target_coords"]
            ego_point = simulation_dict["ego_point"]
            occluders = simulation_dict["occluders"]
            occluded_regions = simulation_dict["occluded_regions"]

            # visualization part
            p_occls = [coords[0] for coords in occlusion_target_coords]
            p_disoccls = [coords[1] for coords in occlusion_target_coords]
            p1s = [occluder[0] for occluder in occluders]
            p2s = [occluder[1] for occluder in occluders]

            plot_simulation_step_6(
                ax=ax,
                p_occls=p_occls,
                p_disoccls=p_disoccls,
                ego_point=ego_point,
                p1s=p1s,
                p2s=p2s,
                occluded_regions=occluded_regions
            )

        except Exception as ex:
            err_count += 1
            logger.exception("\n\nSimulation Failed:\n")
            ax.text(
                sum(ax.get_xlim())/2, sum(ax.get_ylim())/2, "FAILED",
                fontsize=20, c="red", horizontalalignment="center", verticalalignment="center"
            )

        ax.set_title(idx)

        img_path = os.path.join(save_path, f"example_{idx}.png")
        plt.savefig(fname=img_path, format='png', bbox_inches='tight', pad_inches=0)

    print(f"TOTAL AMOUNT OF ERRORS: {err_count} / {n_examples}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-cfg',
        type=os.path.abspath, default=os.path.join(conf.REPO_ROOT, 'config', 'dataset_config.yaml'),
        help='name of the .yaml config file to use for the instantiation of the Dataset.'
    )
    parser.add_argument(
        '--simulator-cfg',
        type=os.path.abspath, required=True,
        help='name of the .yaml config file to use for the parameters of the occlusion simulator.'
    )
    args = parser.parse_args()

    save_simulation_cases(
        dataset_cfg=args.dataset_cfg,
        simulator_cfg=args.simulator_cfg
    )
