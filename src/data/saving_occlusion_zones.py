import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import skgeom as sg
from tqdm import tqdm
from typing import List, Tuple
import src.data.sdd_extract as sdd_extract
import src.occlusion_simulation.visibility as visibility
import src.occlusion_simulation.polygon_generation as poly_gen
import src.visualization.plot_utils as plot_utils
from src.data.sdd_dataloader import StanfordDroneDatasetWithOcclusionSim


def quick_occl_poly(ego_point: Tuple[float, float],
                    occluder: Tuple[np.array, np.array],
                    dist: float) -> sg.Polygon:
    # print(f"{ego_point=}")
    # print(f"{occluder=}")
    occl1 = occluder[0]
    occl2 = occluder[1]
    tri1 = sg.Polygon(np.array([ego_point, occl1, occl2]))
    # bf = time.time()
    if tri1.orientation() == sg.Sign.CLOCKWISE:
        tri1.reverse_orientation()
    # dur = time.time() - bf
    # print(f"{tri1=}")

    p1 = occl1 - ego_point
    p1 *= dist / np.linalg.norm(p1)
    p1 += ego_point

    p2 = occl2 - ego_point
    p2 *= dist / np.linalg.norm(p2)
    p2 += ego_point

    # bf = time.time()
    tri2 = sg.Polygon(np.array([ego_point, p1, p2]))
    if tri2.orientation() == sg.Sign.CLOCKWISE:
        tri2.reverse_orientation()
    # dur += time.time() - bf
    # print(f"{tri2=}")
    # print(f"{dur=}")

    poly_diff = sg.boolean_set.difference(tri2, tri1)

    # fig, axes = plt.subplots(1, 3)
    # plot_utils.plot_sg_polygon(ax=axes[0], poly=tri1)
    # plot_utils.plot_sg_polygon(ax=axes[1], poly=tri2)
    # plot_utils.plot_sg_polygon(ax=axes[2], poly=poly_diff[0])
    # plt.show()
    return poly_diff[0].outer_boundary()


def reformat_occluder_columns(overwrite: bool = False):
    config = sdd_extract.get_config("config")
    pickle_path = os.path.abspath(os.path.join(sdd_extract.REPO_ROOT, config["dataset"]["pickle_path"], "ce4ae1bc-5369-464c-9556-3b33a0178f98"))

    print(f"{pickle_path=}")
    print(f"{os.path.exists(pickle_path)=}")
    sim_folders = [path for path in os.scandir(pickle_path) if path.is_dir()]

    for dir in sim_folders:
        sim_pkl_path = os.path.join(dir, "simulation.pickle")
        print(f"{sim_pkl_path=}")
        print(f"{os.path.exists(sim_pkl_path)=}")
        with open(os.path.abspath(sim_pkl_path), "rb") as f:
            occlusion_table = pickle.load(f)

            print(f"{occlusion_table.columns.values=}")
            print(f"BEFORE: {occlusion_table.head()}")
            occlusion_table.occluders = occlusion_table.occluders.apply(lambda x: [(elem[0].flatten(), elem[1].flatten()) for elem in x])
            print(f"AFTER: {occlusion_table.head()}")

        if overwrite:
            with open(os.path.abspath(sim_pkl_path), "wb") as f:
                pickle.dump(occlusion_table, f)


def create_obs_masks():
    config = sdd_extract.get_config("config")

    dataset = StanfordDroneDatasetWithOcclusionSim(config_dict=config)
    print(f"{len(dataset)=}")
    print(f"{dataset.frames.head()=}")
    print(f"{dataset.lookuptable.head()=}")
    print(f"{dataset.occlusion_table.head()=}")

    pickle_path = os.path.abspath(os.path.join(sdd_extract.REPO_ROOT, config["dataset"]["pickle_path"], dataset.pickle_id))
    print(f"{pickle_path=}")
    print(f"{os.path.exists(pickle_path)=}")
    sim_folders = [path for path in os.scandir(pickle_path) if path.is_dir()]

    simulation_id = dataset.__getitem__(0)["sim_id"]

    occlusion_masks = []

    for idx in tqdm(range(len(dataset))):
        instance_dict = dataset.__getitem__(idx)

        if instance_dict["sim_id"] != simulation_id:
            # dump the pickle
            occl_path = os.path.join(pickle_path, simulation_id, "simulation_occlusions.pickle")
            print(f"Saving occlusion masks to:\n{occl_path}")
            with open(occl_path, "wb") as f:
                pickle.dump(occlusion_masks, f)

            # empty the list
            occlusion_masks = []

        ego_visipoly = visibility.compute_visibility_polygon(
            ego_point=instance_dict["ego_point"],
            occluders=instance_dict["occluders"],
            boundary=poly_gen.default_rectangle(corner_coords=(instance_dict['scene_image'].shape[:2]))
        )
        occlusion_masks.append(visibility.occlusion_masks(agents=instance_dict["agents"],
                                                          time_window=instance_dict["full_window"],
                                                          ego_visipoly=ego_visipoly))
        simulation_id = instance_dict["sim_id"]

    # dump the pickle
    occl_path = os.path.join(pickle_path, simulation_id, "simulation_occlusions.pickle")
    print(f"Saving occlusion masks to:\n{occl_path}")
    with open(occl_path, "wb") as f:
        pickle.dump(occlusion_masks, f)


if __name__ == '__main__':
    create_obs_masks()

