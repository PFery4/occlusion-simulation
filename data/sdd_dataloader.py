import os.path
import pandas as pd
from torch.utils.data import Dataset
import sdd_extract
import sdd_data_processing
import numpy as np
import cv2
import torchvision.transforms
import torch

class StanfordDroneDataset(Dataset):

    def __init__(self, config):
        self.root = config["dataset"]["path"]
        self.orig_fps = config["dataset"]["fps"]
        self.fps = config["hyperparameters"]["fps"]
        self.T_obs = config["hyperparameters"]["T_obs"]
        self.T_pred = config["hyperparameters"]["T_pred"]

        # to convert cv2 image to torch tensor
        self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        frames = []
        scene_keys = []

        # the lookuptable is a dataframe separate from the dataframe containing all trajectory data.
        # each row fully describes a complete training instance, with its corresponding video/scene name, and timestep.
        self.lookuptable = pd.DataFrame(columns=["scene/video", "timestep"])

        # read the csv's for all videos
        # hopefully it is manageable within memory
        for scene_name in os.scandir(os.path.join(self.root, "annotations")):
            for video_name in os.scandir(os.path.realpath(scene_name)):
                scene_key = f"{scene_name.name}/{video_name.name}"
                annot_file_path = os.path.join(os.path.realpath(video_name), "annotations.txt")
                print(annot_file_path)
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
                # annot_df["scene_name"] = scene_name.name
                # annot_df["video_name"] = video_name.name
                # annot_df["window"] = False
                # annot_df["scene/video"] = f"{scene_name.name}/{video_name.name}"

                # print(annot_df[["Id", "frame"]])
                # print(scene_name.name, video_name.name)
                timesteps = sorted(annot_df["frame"].unique())
                # print(timesteps)
                # assert (np.array(timesteps[:-1]) + int(self.orig_fps//self.fps) == np.array(timesteps[1:])).all()

                # We are interested in time windows of [-T_obs : T_pred].
                # But only windows which contain at least one agent for us to be able to predict, otherwise there's
                # nothing useful in a training instance.
                for idx, timestep in enumerate(timesteps[:-(self.T_obs + self.T_pred - 1)]):
                    window = np.array(timesteps[idx: idx + self.T_obs + self.T_pred - 1])
                    # print(idx, window)

                    # some timesteps are inexistant in the dataframe, due to the absence of any annotations at those
                    # points in time. we check that we can have a complete window. (note that some agents might not have
                    # all their timesteps present within the window)
                    if (window[:-1] + int(self.orig_fps//self.fps) == window[1:]).all():
                        self.lookuptable.loc[len(self.lookuptable)] = {"scene/video": scene_key, "timestep": timestep}
                        # self.lookuptable.append({"scene/video": scene_key, "timestep": timestep}, ignore_index=True)
                        # annot_df.loc[annot_df["frame"] == timestep, ["window"]] = True
                    # else:
                    #     print(f"FOUND WRONG WINDOW; {idx} ; {window}")

                frames.append(annot_df)
                scene_keys.append(scene_key)

        self.frames = pd.concat(frames, keys=scene_keys)

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

        # extract the past and futures of present agents at the window starting at the given timestep
        # generate a window of the timesteps we are interested in extracting from the scene dataset
        window = np.arange(self.T_obs + self.T_pred) * int(self.orig_fps // self.fps) + lookup["timestep"]
        # generate subdataframe of the scene with only the timesteps present within the window
        small_df = self.frames.loc[lookup["scene/video"]][self.frames.loc[lookup["scene/video"]]["frame"].isin(window)]

        labels = []
        pasts = []
        futures = []

        for agent_id in small_df["Id"].unique():
            # label of the agent of interest
            label = small_df[small_df["Id"] == agent_id].iloc[0].loc["label"]

            # empty sequence
            sequence = np.empty((self.T_obs + self.T_pred, 2))
            sequence.fill(np.nan)

            # boolean array that indicates for the agent of interest which timesteps contain an observed measurement
            observed_timesteps = np.in1d(window, small_df.loc[small_df["Id"] == agent_id, ["frame"]].values.flatten())

            # filling the trajectory sequence according to the observed_timesteps boolean array
            sequence[observed_timesteps, 0] = small_df.loc[small_df["Id"] == agent_id, ["x"]].values.flatten()
            sequence[observed_timesteps, 1] = small_df.loc[small_df["Id"] == agent_id, ["y"]].values.flatten()

            # appending data to lists of data
            labels.append(label)
            pasts.append(torch.from_numpy(sequence[:self.T_obs, :]))
            futures.append(torch.from_numpy(sequence[self.T_obs:, :]))

        return pasts, futures, labels, image_tensor


if __name__ == '__main__':

    config = sdd_extract.get_config()

    dataset = StanfordDroneDataset(config=config)

    print(dataset.frames)
    print(dataset.lookuptable)
    print(len(dataset))
    dataset.__getitem__(0)
    dataset.__getitem__(2)
    dataset.__getitem__(-1)
