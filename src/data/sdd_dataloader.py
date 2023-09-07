import os.path
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import cv2
import torchvision.transforms
import pickle
import json
import uuid
from src.data.sdd_agent import StanfordDroneAgent
import src.data.config as conf
import src.data.sdd_data_processing as sdd_data_processing


class StanfordDroneDataset(Dataset):

    def __init__(self, config_dict):
        self.root = config_dict["dataset"]["path"]
        datasets_path = os.path.abspath(os.path.join(conf.REPO_ROOT, config_dict["dataset"]["pickle_path"]))
        self.pickle_id = config_dict["dataset"].get("pickle_id", None)

        self.orig_fps = None
        self.fps = None
        self.T_obs = None
        self.T_pred = None
        self.min_n = None
        self.agent_classes = None
        self.other_agents = None

        # to convert cv2 image to torch tensor
        self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        if self.pickle_id:
            dataset_path = os.path.join(datasets_path, self.pickle_id)
            assert os.path.exists(dataset_path), f"ERROR: pickle_id does not exist:\n{self.pickle_id}"
            print(f"Loading dataloader from:\n{dataset_path}")
            json_path = os.path.join(dataset_path, "dataset_parameters.json")
            pickle_path = os.path.join(dataset_path, "dataset.pickle")
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

        else:
            print("No pickle file found/provided, loading from raw dataset and performing preprocessing")

            self.orig_fps = config_dict["dataset"]["fps"]
            self.fps = config_dict["hyperparameters"]["fps"]
            self.T_obs = config_dict["hyperparameters"]["T_obs"]
            self.T_pred = config_dict["hyperparameters"]["T_pred"]
            self.min_n = config_dict["hyperparameters"]["min_N_agents"]
            self.agent_classes = config_dict["hyperparameters"]["agent_types"]
            self.other_agents = config_dict["hyperparameters"]["other_agents"]

            # the lookuptable is a dataframe separate from the dataframe containing all trajectory data.
            # each row fully describes a complete training instance,
            # with its corresponding video/scene name, and timestep.
            self.lookuptable = pd.DataFrame(columns=["scene", "video", "timestep", "targets", "others"])

            frames = []

            # read the csv's for all videos
            # hopefully it is manageable within memory
            for scene_name in os.scandir(os.path.join(self.root, "annotations")):
                for video_name in os.scandir(os.path.realpath(scene_name)):
                    # scene_key = f"{scene_name.name}/{video_name.name}"
                    annot_file_path = os.path.join(os.path.realpath(video_name), "annotations.txt")
                    print(f"Processing: {annot_file_path}")
                    assert os.path.exists(annot_file_path)
                    annot_df = sdd_data_processing.pd_df_from(annotation_filepath=annot_file_path)

                    # perform preprocessing steps
                    if self.other_agents == "OUT":
                        annot_df = annot_df[annot_df["label"].isin(self.agent_classes)]
                    annot_df = sdd_data_processing.perform_preprocessing_pipeline(annot_df=annot_df,
                                                                                  target_fps=self.fps,
                                                                                  orig_fps=self.orig_fps)

                    timesteps = sorted(annot_df["frame"].unique())

                    # We are interested in time windows of [-T_obs : T_pred].
                    # But only windows which contain at least one agent for us to be able to predict,
                    # otherwise the training instance is useless.
                    for t_idx, timestep in enumerate(timesteps[:-(self.T_obs + self.T_pred)]):
                        window = np.array(timesteps[t_idx: t_idx + self.T_obs + self.T_pred])
                        # print(idx, window)

                        # some timesteps are inexistant in the dataframe, due to the absence of any annotations at those
                        # points in time. we check that we can have a complete window.
                        # (note that while a complete window means we have at least one observation for each timestep,
                        # some agents might not have all their timesteps observed)
                        window_is_complete = (window[:-1] + int(self.orig_fps//self.fps) == window[1:]).all()

                        if window_is_complete:
                            mini_df = annot_df[annot_df["frame"].isin(window)]
                            all_present_agents = mini_df["Id"].unique()
                            candidate_target_agents = mini_df[mini_df["label"].isin(self.agent_classes)]["Id"].unique()

                            # we are only interested in windows with at least 1 fully described trajectory, that is,
                            # with at least one agent who's observed at all timesteps in the window
                            target_agents = [agent_id for agent_id in candidate_target_agents if
                                             (mini_df["Id"].values == agent_id).sum() == self.T_obs + self.T_pred]
                            other_agents = [agent_id for agent_id in all_present_agents if
                                              agent_id not in target_agents]

                            if len(target_agents) >= self.min_n:
                                self.lookuptable.loc[len(self.lookuptable)] = {
                                    "scene": scene_name.name,
                                    "video": video_name.name,
                                    "timestep": timestep,
                                    "targets": target_agents,
                                    "others": other_agents
                                }

                    annot_df.insert(0, "scene", scene_name.name, False)
                    annot_df.insert(1, "video", video_name.name, False)

                    frames.append(annot_df)
                    # scene_keys.append(scene_key)

            self.lookuptable.set_index(["scene", "video", "timestep"], inplace=True)
            self.lookuptable.sort_index(inplace=True)
            self.frames = pd.concat(frames)
            self.frames.set_index(["scene", "video"], inplace=True)
            self.frames.sort_index(inplace=True)

            # generate unique file name for our saved pickle and json files
            self.pickle_id = str(uuid.uuid4())
            self.save_data(datasets_path, self.pickle_id)

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

        # todo: maybe consider doing something with partially observed agents in the delivery of the instance_dict
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
        }

        return instance_dict

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

    def find_pickle_id(self, search_dir: str):
        """
        looks for a pickle file with same corresponding metadata as self.metadata_dict()
        :param search_dir: the directory within which to look
        :return: the path to a valid pickle file, if it exists
        """
        from warnings import warn
        warn("This function is no longer used", DeprecationWarning, stacklevel=2)

        json_files = [os.path.join(search_dir, file) for file in os.listdir(search_dir) if file.endswith(".json")]
        json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        for jsonfile in json_files:
            with open(jsonfile) as f:
                if json.load(f) == self.metadata_dict():
                    return f"{os.path.splitext(os.path.basename(jsonfile))[0]}"
        return None

    def save_data(self, path: str, save_name: str):
        """
        saves 'self.frames' and 'self.lookuptable' into a pickle file, and the corresponding parameters as a json
        :return: None
        """
        # create dataset directory
        dataset_path = os.path.join(path, save_name)
        os.mkdir(dataset_path)

        # save lookuptable and frames to a pickle
        with open(os.path.join(dataset_path, f"dataset.pickle"), "wb") as f:
            pickle.dump((self.frames, self.lookuptable), f)

        # save metadata as a json with same file name as pickle file
        with open(os.path.join(dataset_path, f"dataset_parameters.json"), "w", encoding="utf8") as f:
            json.dump(self.metadata_dict(), f, indent=4)

    @staticmethod
    def load_data(filepath: str):
        """
        reads the pickle file with name 'filepath',
        and assigns corresponding values to 'self.frames' and 'self.lookuptable'
        :param filepath: the path of the pickle file to read from
        :return: None
        """
        with open(os.path.join(filepath), "rb") as f:
            return pickle.load(f)


class StanfordDroneDatasetWithOcclusionSim(StanfordDroneDataset):
    def __init__(self, config_dict):
        super(StanfordDroneDatasetWithOcclusionSim, self).__init__(config_dict)

        pickle_path = os.path.abspath(os.path.join(conf.REPO_ROOT, config_dict["dataset"]["pickle_path"]))
        sim_root_dir = os.path.join(pickle_path, self.pickle_id)

        assert os.path.exists(sim_root_dir)
        print(f"Simulation pickle files are stored under:\n{sim_root_dir}")

        sim_folders = [path for path in os.scandir(sim_root_dir) if path.is_dir()]

        occlusion_tables = []
        sim_ids = []

        for folder in sim_folders:
            sim_pkl_path = os.path.join(folder, "simulation.pickle")
            sim_json_path = os.path.join(folder, "simulation.json")

            assert os.path.exists(sim_pkl_path) and os.path.exists(sim_json_path), \
                "Simulation pickle and json files not found, " \
                "please run the occlusion simulation and verify that the corresponding files have been saved"
            print(f"extracting data form: {sim_pkl_path}")

            occlusion_tables.append(self.load_data(sim_pkl_path))
            sim_ids.append(os.path.basename(folder))

        self.occlusion_table = pd.concat(occlusion_tables, keys=sim_ids, names=["sim_id"])
        # print(self.occlusion_table.head())

    def __len__(self) -> int:
        return len(self.occlusion_table)

    def __getitem__(self, idx: int) -> dict:
        # lookup the row in self.occlusion_table
        occlusion_case = self.occlusion_table.iloc[idx]

        sim_id, scene, video, timestep, trial = occlusion_case.name

        # find the corresponding index in self.lookuptable
        lookup_idx = self.find_idx(scene=scene, video=video, timestep=timestep)

        instance_dict = super(StanfordDroneDatasetWithOcclusionSim, self).__getitem__(lookup_idx)

        instance_dict["sim_id"] = sim_id
        instance_dict["trial"] = trial
        instance_dict["ego_point"] = occlusion_case["ego_point"]
        instance_dict["occluders"] = occlusion_case["occluders"]
        instance_dict["target_agent_indices"] = occlusion_case["target_agent_indices"]
        instance_dict["occlusion_windows"] = occlusion_case["occlusion_windows"]

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

    def find_occl_idx(self, sim_id: str, scene: str, video: str, timestep: int, trial: int) -> int:
        video_slice = self.occlusion_table.index.get_loc((sim_id, scene, video, timestep, trial))
        return int(video_slice)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import src.visualization.sdd_visualize as sdd_visualize

    config = conf.get_config("config")

    dataset = StanfordDroneDataset(config_dict=config)
    # dataset = StanfordDroneDatasetWithOcclusionSim(config_dict=config)
    print(f"{len(dataset)=}")

    print(f"{dataset.__class__}.__getitem__() dictionary keys:")
    [print(k) for k in dataset.__getitem__(0).keys()]
    print()

    n_rows = 2
    n_cols = 2

    ##################################################################################################################
    fig, axes = plt.subplots(n_rows, n_cols)
    fig.canvas.manager.set_window_title(f"{dataset.__class__}.__getitem__()")

    idx_samples = np.sort(np.random.randint(0, len(dataset), n_rows * n_cols))

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

    plt.show()
    ##################################################################################################################

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
