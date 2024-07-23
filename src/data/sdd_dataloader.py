import os.path
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import cv2
import torchvision.transforms
import pickle
import json
from src.data.sdd_agent import StanfordDroneAgent
import src.data.config as conf
from typing import Dict, Optional


class StanfordDroneDataset(Dataset):

    def __init__(self, config_dict: Dict, split: Optional[str] = None):
        assert split in ('train', 'val', 'test') or split is None
        self.root = config_dict["dataset"]["path"]
        saved_datasets_path = os.path.abspath(os.path.join(conf.REPO_ROOT, config_dict["dataset"]["pickle_path"]))
        assert os.path.exists(saved_datasets_path), f"ERROR: saved datasets path does not exist:\n{saved_datasets_path}"
        self.pickle_id = config_dict["dataset"]["pickle_id"]
        self.split = split

        self.coord_conv = conf.COORD_CONV

        # to convert cv2 image to torch tensor
        self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.dataset_path = os.path.join(saved_datasets_path, self.pickle_id)
        assert os.path.exists(self.dataset_path), f"ERROR: dataset directory does not exist:\n{self.dataset_path}"
        print(f"Loading dataloader from:\n{self.dataset_path}")
        json_path = os.path.join(self.dataset_path, "dataset_parameters.json")
        pickle_path = os.path.join(self.dataset_path, "dataset.pickle")
        assert os.path.exists(json_path)
        assert os.path.exists(pickle_path)

        with open(json_path) as f:
            json_dict = json.load(f)

        self.orig_fps = json_dict["orig_fps"]
        self.fps = json_dict["fps"]
        self.T_obs = json_dict["T_obs"]
        self.T_pred = json_dict["T_pred"]
        self.min_n = json_dict["min_n"]
        self.agent_classes = json_dict["agent_classes"]
        self.other_agents = json_dict["other_agents"]

        self.frames, self.lookuptable = self.load_data(pickle_path)

        # subsetting the data from the splits we are interested in
        if self.split is not None:
            self.split_subset()

    def __len__(self) -> int:
        return len(self.lookuptable)

    def __getitem__(self, idx: int) -> dict:
        # lookup the row in self.lookuptable
        lookup = self.lookuptable.iloc[idx]
        scene, video, timestep = lookup.name

        # extract the reference image
        image_path = os.path.join(self.root, "annotations", scene, video, "reference.jpg")
        assert os.path.exists(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # np.array [H, W, C]
        # image_tensor = self.img_transform(image)            # torch.tensor [C, H, W]

        # generate a window of the timesteps we are interested in extracting from the scene dataset
        window = np.arange(self.T_obs + self.T_pred) * int(self.orig_fps // self.fps) + timestep
        # generate subdataframe of the specific video
        scenevideo_df = self.frames.loc[(scene, video)]

        agents = [StanfordDroneAgent(scenevideo_df[scenevideo_df["Id"] == agent_id]) for agent_id in lookup["targets"]]

        # TODO: maybe consider doing something with partially observed agents in the delivery of the instance_dict
        # # extract the trajectories of partially observed agents
        # for agent_id in lookup["present"]:
        #     # label of the agent of interest
        #     label = mini_df[mini_df["Id"] == agent_id].iloc[0].loc["label"]
        #
        #     # empty sequence
        #     sequence = np.empty((self.T_obs + self.T_pred, 2))
        #     sequence.fill(np.nan)
        #
        #     # boolean array that indicates for the agent of interest which timesteps contain an observed measurement
        #     observed_timesteps = np.in1d(window, mini_df.loc[mini_df["Id"] == agent_id, ["frame"]].values.flatten())
        #
        #     # filling the trajectory sequence according to the observed_timesteps boolean array
        #     sequence[observed_timesteps, 0] = mini_df.loc[mini_df["Id"] == agent_id, ["x"]].values.flatten()
        #     sequence[observed_timesteps, 1] = mini_df.loc[mini_df["Id"] == agent_id, ["y"]].values.flatten()
        #
        #     # appending data to lists of data
        #     agent_ids.append(agent_id)
        #     labels.append(label)
        #     pasts.append(torch.from_numpy(sequence[:self.T_obs, :]))
        #     futures.append(torch.from_numpy(sequence[self.T_obs:, :]))
        #     is_fully_observed.append(False)

        instance_dict = {
            "idx": idx,
            "scene": scene,
            "video": video,
            "timestep": timestep,
            "agents": agents,
            "past_window": window[:self.T_obs],
            "future_window": window[self.T_obs:],
            "full_window": window,
            "scene_image": image,
            "px/m": self.coord_conv.loc[scene, video]['px/m'],
            "m/px": self.coord_conv.loc[scene, video]['m/px']
        }

        return instance_dict

    def split_subset(self):
        scene_video_indices_lookuptable = self.lookuptable.index.droplevel(2)
        scene_video_indices_frames = self.frames.index
        split_indices = scene_video_indices_lookuptable.unique()[
            np.array(conf.SCENE_SPLIT) == self.split
            ]
        self.lookuptable = self.lookuptable[scene_video_indices_lookuptable.isin(split_indices)]
        self.frames = self.frames[scene_video_indices_frames.isin(split_indices)]
        assert (self.lookuptable.index.droplevel(2).unique() == self.frames.index.unique()).all()

    def find_idx(self, scene: str, video: str, timestep: int) -> int:
        video_slice = self.lookuptable.index.get_loc((scene, video, timestep))
        return int(video_slice)

    def metadata_dict(self) -> dict:
        metadata_dict = {
            "orig_fps": self.orig_fps,
            "fps": self.fps,
            "T_obs": self.T_obs,
            "T_pred": self.T_pred,
            "min_n": self.min_n,
            "agent_classes": self.agent_classes,
            "other_agents": self.other_agents
        }
        return metadata_dict

    @staticmethod
    def load_data(filepath: str):
        """ reads the pickle file with name 'filepath', """
        with open(os.path.join(filepath), "rb") as f:
            return pickle.load(f)


class StanfordDroneDatasetWithOcclusionSim(StanfordDroneDataset):
    def __init__(self, config_dict: Dict, split: Optional[str] = None):
        super().__init__(config_dict=config_dict, split=split)

        assert os.path.exists(self.dataset_path), f"ERROR: Path does not exist: {self.dataset_path}"
        print(f"Extracting simulation pickle files from:\n{self.dataset_path}")

        sim_ids = config_dict["occlusion_dataset"]["sim_ids"]
        n_trials = []
        occlusion_tables = []

        for sim_id in sim_ids:
            sim_folder = os.path.join(self.dataset_path, sim_id)
            assert os.path.exists(sim_folder), f"ERROR: simulation folder does not exist:\n{sim_folder}"
            sim_pkl_path = os.path.join(sim_folder, "simulation.pickle")
            sim_json_path = os.path.join(sim_folder, "simulation_parameters.json")
            assert os.path.exists(sim_pkl_path) and os.path.exists(sim_json_path), \
                "Simulation pickle and/or json files not found, " \
                "please run the occlusion simulation and verify that the corresponding files have been saved:\n" \
                f"{sim_pkl_path}\n" \
                f"{sim_json_path}"
            print(f"extracting data form:\n{sim_pkl_path}\n{sim_json_path}")
            occlusion_tables.append(self.load_data(sim_pkl_path))
            with open(sim_json_path, 'r') as f:
                sim_params = json.load(f)
            n_trials.append(sim_params["simulations_per_instance"])

        self.occlusion_table = pd.concat(occlusion_tables, keys=sim_ids, names=["sim_id"])

        # extracting every case for which no occlusion simulation is available
        all_indices = [
            (sim_id, *index, n-1)
            for sim_id, n in zip(sim_ids, n_trials)
            for index in self.lookuptable.index
        ]
        empty_indices = pd.Index(list(set(all_indices).difference(self.occlusion_table.index))).sort_values()
        keep = (empty_indices.droplevel(4)[:-1] != empty_indices.droplevel(4)[1:])
        keep = np.concatenate([[True], keep])
        empty_indices = empty_indices[keep]

        self.occlusion_table = self.occlusion_table.reindex(
            [*self.occlusion_table.index, *empty_indices]
        ).fillna(value=np.nan)
        self.occlusion_table.sort_index(inplace=True)

        if self.split is not None:
            self.split_subset_occlusion_table()

        self.empty_cases = self.occlusion_table['ego_point'].isna().sum()
        self.occlusion_cases = len(self.occlusion_table) - self.empty_cases

        # adding to each occlusion row the corresponding index of self.lookuptable
        self.occlusion_table['lookup_idx'] = self.lookuptable.index.get_indexer(
            self.occlusion_table.index.droplevel(['sim_id', 'trial'])
        )

        self.print_occlusion_summary()

    def __len__(self) -> int:
        return len(self.occlusion_table)

    def __getitem__(self, idx: int) -> dict:
        # lookup the row in self.occlusion_table
        occlusion_case = self.occlusion_table.iloc[idx]

        sim_id, scene, video, timestep, trial = occlusion_case.name

        # find the corresponding index in self.lookuptable
        lookup_idx = occlusion_case['lookup_idx']

        instance_dict = super(StanfordDroneDatasetWithOcclusionSim, self).__getitem__(lookup_idx)

        instance_dict["sim_id"] = sim_id
        instance_dict["trial"] = trial
        instance_dict["ego_point"] = occlusion_case["ego_point"]
        instance_dict["occluders"] = occlusion_case["occluders"]
        instance_dict["target_agent_indices"] = occlusion_case["target_agent_indices"]
        instance_dict["occlusion_windows"] = occlusion_case["occlusion_windows"]

        # print(f"{instance_dict.items()=}")

        # ego_visipoly = visibility.compute_visibility_polygon(
        #     ego_point=instance_dict["ego_point"],
        #     occluders=instance_dict["occluders"],
        #     boundary=poly_gen.default_rectangle(corner_coords=(instance_dict['scene_image'].shape[:2]))
        # )
        #
        # instance_dict["full_window_occlusion_masks"] = visibility.occlusion_masks(
        #     agents=instance_dict["agents"],
        #     time_window=instance_dict["full_window"],
        #     ego_visipoly=ego_visipoly
        # )
        # Maybe load occlusion masks here

        return instance_dict

    def print_occlusion_summary(self) -> None:
        print(f"Total Number of instances:\t\t\t\t\t\t\t{len(self)}")
        print(f"Number of instances with an occlusion simulation:\t{self.occlusion_cases} ({self.occlusion_cases / len(self) * 100:.2f}%)")
        print(f"Number of instances without occlusion simulation:\t{self.empty_cases} ({self.empty_cases / len(self) * 100:.2f}%)")

    def split_subset_occlusion_table(self) -> None:
        scene_video_indices_occlusion_table = self.occlusion_table.index.droplevel([0, 3, 4])
        split_indices = self.lookuptable.index.droplevel(2).unique()
        self.occlusion_table = self.occlusion_table[scene_video_indices_occlusion_table.isin(split_indices)]
        assert (self.lookuptable.index.droplevel(2).unique() ==
                self.occlusion_table.index.droplevel([0, 3, 4]).unique()).all()

    def find_occl_idx(self, sim_id: str, scene: str, video: str, timestep: int, trial: int) -> int:
        video_slice = self.occlusion_table.index.get_loc((sim_id, scene, video, timestep, trial))
        return int(video_slice)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import src.visualization.sdd_visualize as sdd_visualize

    config = conf.get_config("config")

    # dataset = StanfordDroneDataset(config_dict=config, split='train')
    dataset = StanfordDroneDatasetWithOcclusionSim(config_dict=config, split=None)
    print(f"{len(dataset)=}")

    # print(f"{dataset.__class__}.__getitem__() dictionary keys:")
    # [print(k) for k in dataset.__getitem__(0).keys()]

    n_rows = 2
    n_cols = 2

    ##################################################################################################################
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.canvas.manager.set_window_title(f"{dataset.__class__}.__getitem__()")

    idx_samples = np.sort(np.random.randint(0, len(dataset), n_rows * n_cols))
    idx_samples = [0, 1, 2, 3]

    print(f"{len(dataset)=}")

    print(f"visualizing the following randomly chosen samples: {idx_samples}")
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
    ##################################################################################################################

    # idx_samples = [42578, 62777, 79908, 41933, 90340, 80810, 79397]
    # for idx in idx_samples:
    #     fig, ax = plt.subplots(1, 1)
    #     instance_dict = dataset.__getitem__(idx)
    #
    #     ax.title.set_text(idx)
    #     sdd_visualize.visualize_training_instance(
    #         draw_ax=ax, instance_dict=instance_dict
    #     )

    # print(dataset.frames.columns)
    #
    # indices = [np.random.randint(0, len(dataset))]
    # indices = [33640, 33641, 33642, 33643, 33644, 33645]
    # print(f"{indices=}")
    #
    # [print(dataset.__getitem__(idx)["scene"]) for idx in indices]
    # [print(dataset.__getitem__(idx)["video"]) for idx in indices]
    # [print(dataset.__getitem__(idx)["agent_ids"]) for idx in indices]
    # [print(dataset.__getitem__(idx)["timestep"]) for idx in indices]

    ##################################################################################################################
    plt.show()
