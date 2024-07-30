import argparse
import matplotlib.pyplot as plt
import os.path

import src.data.config as conf
from src.data.sdd_dataloader import StanfordDroneDataset
from src.occlusion_simulation.simple_occlusion import simulate_occlusions
from src.visualization import sdd_visualize
from src.visualization.simulation_visualize import visualize_occlusion_simulation, visualize_random_simulation_samples


def show_simulation(
        dataset_cfg: str,
        simulator_cfg: str,
        instance_idx: int
):
    dataset_config = conf.get_config(dataset_cfg)
    simulator_config = conf.get_config(simulator_cfg)

    dataset = StanfordDroneDataset(config=dataset_config)

    # showing the simulation process of some desired instance
    print(f"dataset.__getitem__({instance_idx})")
    instance_dict = dataset.__getitem__(instance_idx)

    fig, ax = plt.subplots()
    sdd_visualize.draw_map_numpy(draw_ax=ax, scene_image=instance_dict["scene_image"])
    sdd_visualize.visualize_training_instance(draw_ax=ax, instance_dict=instance_dict)

    # time_polygon_generation(instance_dict=instance_dict, n_iterations=100000)
    sim_params = simulator_config
    img = instance_dict["scene_image"]
    agents = instance_dict["agents"]
    past_window = instance_dict["past_window"]
    future_window = instance_dict["future_window"]

    simulation_outputs = simulate_occlusions(
        config=sim_params,
        image_res=tuple(img.shape[:2]),
        agents=agents,
        past_window=past_window,
        future_window=future_window
    )
    visualize_occlusion_simulation(instance_dict, simulation_outputs)

    # Showing the simulation outputs of some random instances
    visualize_random_simulation_samples(dataset, simulator_config, 2, 2)


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
    parser.add_argument('--instance-idx', type=int, default=7592,
                        help='instance index to use as the example to execute the simulator on.')
    args = parser.parse_args()

    show_simulation(
        dataset_cfg=args.dataset_cfg,
        simulator_cfg=args.simulator_cfg,
        instance_idx=args.instance_idx
    )
