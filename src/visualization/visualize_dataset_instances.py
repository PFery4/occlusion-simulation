import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

import src.data.config as conf
from src.data.sdd_dataloader import StanfordDroneDataset, StanfordDroneDatasetWithOcclusionSim
import src.visualization.sdd_visualize as sdd_visualize


dataset_classes = {
    'base': StanfordDroneDataset,
    'sim': StanfordDroneDatasetWithOcclusionSim
}


def visualize_dataset_instances(
        dataset_cfg: str,
        dataset_class: str = 'base',
        split: str = None
):
    config = conf.get_config(dataset_cfg)

    dataset = dataset_classes[dataset_class](config=config, split=split)
    print(f"{len(dataset)=}")

    n_rows = 2
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols)
    fig.canvas.manager.set_window_title(f"{dataset.__class__}.__getitem__()")

    idx_samples = np.sort(np.random.randint(0, len(dataset), n_rows * n_cols))

    print(f"{len(dataset)=}")

    print(f"visualizing the following selected samples: {idx_samples}")
    for ax_k, idx in enumerate(idx_samples):

        ax_x, ax_y = ax_k // n_cols, ax_k % n_cols
        ax = axes[ax_x, ax_y] if n_rows != 1 or n_cols != 1 else axes

        before = time.time()
        instance_dict = dataset.__getitem__(idx)
        print(f"getitem({idx}) took {time.time() - before} s")

        ax.title.set_text(idx)
        sdd_visualize.visualize_training_instance(
            draw_ax=ax, instance_dict=instance_dict
        )

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True,
                        help='name of the .yaml config file to use for instantiating the dataset.')
    parser.add_argument('--dataset-class', type=str, default='base',
                        help='[\'base\' | \'sim\'], whether to instantiate a dataset with/without occlusions.')
    parser.add_argument('--split', type=str, default=None,
                        help='[\'train\' | \'test\' | \'val\'], the dataset split to use.'
                             'By default, no split applied.')
    args = parser.parse_args()

    visualize_dataset_instances(
        dataset_cfg=args.cfg,
        dataset_class=args.dataset_class,
        split=args.split
    )
