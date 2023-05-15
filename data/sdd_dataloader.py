import os.path
import pandas as pd
from torch.utils.data import Dataset
import sdd_extract
import sdd_data_processing
import sdd_visualize
import numpy as np
import cv2
import torchvision.transforms
import torch
import matplotlib.pyplot as plt
import pickle
import json
import uuid
import time


class StanfordDroneDataset(Dataset):

    def __init__(self, config_dict):
        self.root = config_dict["dataset"]["path"]
        self.orig_fps = config_dict["dataset"]["fps"]
        self.fps = config_dict["hyperparameters"]["fps"]
        self.T_obs = config_dict["hyperparameters"]["T_obs"]
        self.T_pred = config_dict["hyperparameters"]["T_pred"]

        # to convert cv2 image to torch tensor
        self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        found_path = self.find_pickle(config_dict["dataset"]["pickle_path"])
        if found_path:
            print(f"Loading dataloader from:\n{found_path}")
            self.frames, self.lookuptable = self.load_data(found_path)
        else:
            print("No pickle file found, loading from raw dataset and performing preprocessing")
            # the lookuptable is a dataframe separate from the dataframe containing all trajectory data.
            # each row fully describes a complete training instance,
            # with its corresponding video/scene name, and timestep.
            self.lookuptable = pd.DataFrame(columns=["scene/video", "timestep", "full_obs", "present"])

            frames = []
            scene_keys = []

            # read the csv's for all videos
            # hopefully it is manageable within memory
            for scene_name in os.scandir(os.path.join(self.root, "annotations")):
                for video_name in os.scandir(os.path.realpath(scene_name)):
                    scene_key = f"{scene_name.name}/{video_name.name}"
                    annot_file_path = os.path.join(os.path.realpath(video_name), "annotations.txt")
                    print(f"Processing: {annot_file_path}")
                    assert os.path.exists(annot_file_path)

                    annot_df = sdd_extract.pd_df_from(annotation_filepath=annot_file_path)

                    # perform preprocessing steps when reading the data
                    annot_df = sdd_data_processing.bool_columns_in(annot_df)
                    annot_df = sdd_data_processing.completely_lost_trajs_removed_from(annot_df)
                    annot_df = sdd_data_processing.xy_columns_in(annot_df)
                    annot_df = sdd_data_processing.keep_masks_in(annot_df)
                    annot_df = annot_df[annot_df["keep"]]
                    annot_df = sdd_data_processing.subsample_timesteps_from(
                        annot_df, target_fps=self.fps, orig_fps=self.orig_fps
                    )

                    timesteps = sorted(annot_df["frame"].unique())

                    # We are interested in time windows of [-T_obs : T_pred].
                    # But only windows which contain at least one agent for us to be able to predict,
                    # otherwise the training instance is useless.
                    for t_idx, timestep in enumerate(timesteps[:-(self.T_obs + self.T_pred)]):
                        window = np.array(timesteps[t_idx: t_idx + self.T_obs + self.T_pred])
                        # print(idx, window)

                        # some timesteps are inexistant in the dataframe, due to the absence of any annotations at those
                        # points in time. we check that we can have a complete window.
                        # (note that with this criterion validates,
                        # some agents might not have all their timesteps present within the window)
                        window_is_continuous = (window[:-1] + int(self.orig_fps//self.fps) == window[1:]).all()

                        if window_is_continuous:
                            mini_df = annot_df[annot_df["frame"].isin(window)]
                            all_present_agents = mini_df["Id"].unique()

                            # we are only interested in windows with at least 1 fully described trajectory, that is,
                            # with at least one agent who's observed at all timesteps in the window
                            fully_observed_agents = [agent_id for agent_id in all_present_agents if
                                                     (mini_df["Id"].values == agent_id).sum() ==
                                                     self.T_obs + self.T_pred]
                            present_agents = [agent_id for agent_id in all_present_agents if
                                              agent_id not in fully_observed_agents]

                            if fully_observed_agents:
                                self.lookuptable.loc[len(self.lookuptable)] = {
                                    "scene/video": scene_key,
                                    "timestep": timestep,
                                    "full_obs": fully_observed_agents,
                                    "present": present_agents
                                }

                    frames.append(annot_df)
                    scene_keys.append(scene_key)

            self.frames = pd.concat(frames, keys=scene_keys)

            self.save_data(config_dict["dataset"]["pickle_path"])

    def __len__(self):
        return len(self.lookuptable)

    def __getitem__(self, idx):
        # lookup the row in self.lookuptable
        lookup = self.lookuptable.iloc[idx]

        # extract the reference image
        scene, video = lookup["scene/video"].split("/", 1)
        image_path = os.path.join(self.root, "annotations", scene, video, "reference.jpg")
        assert os.path.exists(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.img_transform(image)

        # generate a window of the timesteps we are interested in extracting from the scene dataset
        window = np.arange(self.T_obs + self.T_pred) * int(self.orig_fps // self.fps) + lookup["timestep"]
        # generate subdataframe of the scene with only the timesteps present within the window
        mini_df = self.frames.loc[lookup["scene/video"]][self.frames.loc[lookup["scene/video"]]["frame"].isin(window)]

        # extract the trajectories of fully observed agents
        labels = []
        pasts = []
        futures = []

        for agent_id in lookup["full_obs"]:
            # label of the agent of interest
            label = mini_df[mini_df["Id"] == agent_id].iloc[0].loc["label"]

            # empty sequence
            sequence = np.empty((self.T_obs + self.T_pred, 2))
            sequence.fill(np.nan)

            sequence[:, 0] = mini_df.loc[mini_df["Id"] == agent_id, ["x"]].values.flatten()
            sequence[:, 1] = mini_df.loc[mini_df["Id"] == agent_id, ["y"]].values.flatten()

            assert not np.isnan(sequence).any()

            labels.append(label)
            pasts.append(torch.from_numpy(sequence[:self.T_obs, :]))
            futures.append(torch.from_numpy(sequence[self.T_obs:, :]))

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
        #     labels.append(label)
        #     pasts.append(torch.from_numpy(sequence[:self.T_obs, :]))
        #     futures.append(torch.from_numpy(sequence[self.T_obs:, :]))

        return pasts, futures, labels, image_tensor

    def metadata_dict(self):
        metadata_dict = {
            "root": self.root,
            "orig_fps": self.orig_fps,
            "fps": self.fps,
            "T_obs": self.T_obs,
            "T_pred": self.T_pred
        }
        return metadata_dict

    def find_pickle(self, search_dir):
        """
        looks for a pickle file with same corresponding metadata as self.metadata_dict()
        :param search_dir: the directory within which to look
        :return: the path to a valid pickle file, if it exists
        """
        json_files = [os.path.join(search_dir, file) for file in os.listdir(search_dir) if file.endswith(".json")]
        json_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        for jsonfile in json_files:
            with open(jsonfile) as f:
                if json.load(f) == self.metadata_dict():
                    return f"{os.path.splitext(jsonfile)[0]}.pickle"
        return None

    def save_data(self, path):
        """
        saves self.frames and self.lookuptable into a pickle file, and the corresponding parameters as a json
        :return: None
        """
        # generate unique file name for our saved pickle and json files
        save_name = str(uuid.uuid4())

        # save lookuptable and frames to a pickle
        with open(os.path.join(path, f"{save_name}.pickle"), "wb") as f:
            pickle.dump((self.frames, self.lookuptable), f)

        # save metadata as a json with same file name as pickle file
        with open(os.path.join(path, f"{save_name}.json"), "w", encoding="utf8") as f:
            json.dump(self.metadata_dict(), f, indent=0)

    def load_data(self, filepath):
        """
        reads the pickle file with name 'filepath', and assigns corresponding values to self.frames and self.lookuptable
        :param filepath: the path of the pickle file to read from
        :return: None
        """
        with open(os.path.join(filepath), "rb") as f:
            return pickle.load(f)


if __name__ == '__main__':

    config = sdd_extract.get_config()

    before = time.time()
    dataset = StanfordDroneDataset(config_dict=config)
    after = time.time()
    print(after - before, "seconds for dataset instantiation")

    print(dataset.lookuptable)
    print(f"{len(dataset)=}")

    fig, axes = plt.subplots(4, 4)
    fig.canvas.manager.set_window_title(f"StanfordDroneDataset.__getitem__()")

    idx_samples = np.sort(np.random.randint(0, len(dataset), 16))

    print(idx_samples)
    for ax_k, idx in enumerate(idx_samples):

        ax_x, ax_y = ax_k // 4, ax_k % 4

        before = time.time()
        pasts, futures, labels, image_tensor = dataset.__getitem__(idx)
        after = time.time()
        print(f"getitem({idx}) took {after - before} seconds")

        axes[ax_x, ax_y].title.set_text(idx)
        sdd_visualize.visualize_training_instance(
            draw_ax=axes[ax_x, ax_y], pasts=pasts, futures=futures, labels=labels, image_tensor=image_tensor
        )

    plt.show()
