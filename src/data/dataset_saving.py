import argparse
import json
import numpy as np
import os.path
import pandas as pd
import pickle
import uuid

from src.data import config
from src.data import sdd_data_processing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None, required=True,
                        help='name of the .yaml config file to use,')
    parser.add_argument('--dirname', type=str, default=None, required=False,
                        help='name of the directory to save processed dataset.')
    args = parser.parse_args()

    if args.dirname is None:
        generated_name = str(uuid.uuid4())
        args.dirname = generated_name
        print(f"No directory name provided for the target, generating a random name instead:\n{generated_name}\n")

    config_dict = config.get_config(args.cfg)

    print("Performing Preprocessing and Saving of Dataset.\n\n")

    dataset_path = config_dict["dataset"]["path"]
    assert os.path.exists(dataset_path), f"ERROR: dataset path does not exist:\n{dataset_path}"
    print(f"Source dataset path:\n{dataset_path}\n")

    # generating a unique directory name for saving our dataset and metadata files
    save_path = os.path.abspath(os.path.join(config.REPO_ROOT, config_dict["dataset"]["pickle_path"]))
    save_dir = os.path.join(save_path, args.dirname)
    assert not os.path.exists(save_dir), f"ERROR: target directory already exists:\n{save_dir}"
    print(f"Target directory:\n{save_dir}\n\n")

    original_fps = config_dict["dataset"]["fps"]
    fps = config_dict["hyperparameters"]["fps"]
    T_obs = config_dict["hyperparameters"]["T_obs"]
    T_pred = config_dict["hyperparameters"]["T_pred"]
    min_n_agents = config_dict["hyperparameters"]["min_N_agents"]
    agent_classes = config_dict["hyperparameters"]["agent_types"]
    other_agents = config_dict["hyperparameters"]["other_agents"]

    # the lookuptable is a dataframe separate from the dataframe containing all trajectory data.
    # each row fully describes a complete training instance,
    # with its corresponding video/scene name, and timestep.
    lookuptable = pd.DataFrame(columns=["scene", "video", "timestep", "targets", "others"])

    frames = []

    print("Starting dataset processing:")
    # read the csv's for all videos (hopefully it is manageable within memory)
    for scene_name in os.scandir(os.path.join(dataset_path, "annotations")):
        for video_name in os.scandir(os.path.realpath(scene_name)):
            annot_file_path = os.path.join(os.path.realpath(video_name), "annotations.txt")
            print(f"Processing: {annot_file_path}")
            assert os.path.exists(annot_file_path)
            annot_df = sdd_data_processing.pd_df_from(annotation_filepath=annot_file_path)

            # perform preprocessing steps
            if other_agents == "OUT":
                annot_df = annot_df[annot_df["label"].isin(agent_classes)]
            annot_df = sdd_data_processing.perform_preprocessing_pipeline(
                annot_df=annot_df, target_fps=fps, orig_fps=original_fps
            )

            timesteps = sorted(annot_df["frame"].unique())

            # We are interested in time windows of [-T_obs : T_pred].
            # But only windows which contain at least one agent for us to be able to predict,
            # otherwise the training instance is useless.
            for t_idx, timestep in enumerate(timesteps[:-(T_obs + T_pred)]):
                window = np.array(timesteps[t_idx: t_idx + T_obs + T_pred])

                # some timesteps are inexistant in the dataframe, due to the absence of any annotations at those
                # points in time. we check that we can have a complete window.
                # (note that while a complete window means we have at least one observation for each timestep,
                # some agents might not have all their timesteps observed)
                window_is_complete = (window[:-1] + int(original_fps // fps) == window[1:]).all()

                if window_is_complete:
                    mini_df = annot_df[annot_df["frame"].isin(window)]
                    all_present_agents = mini_df["Id"].unique()
                    candidate_target_agents = mini_df[mini_df["label"].isin(agent_classes)]["Id"].unique()

                    # we are only interested in windows with at least 1 fully described trajectory, that is,
                    # with at least one agent observed at all timesteps in the window
                    target_agents = [agent_id for agent_id in candidate_target_agents if
                                     (mini_df["Id"].values == agent_id).sum() == T_obs + T_pred]
                    non_target_agents = [agent_id for agent_id in all_present_agents if
                                         agent_id not in target_agents]

                    if len(target_agents) >= min_n_agents:
                        lookuptable.loc[len(lookuptable)] = {
                            "scene": scene_name.name,
                            "video": video_name.name,
                            "timestep": timestep,
                            "targets": target_agents,
                            "others": non_target_agents
                        }

            annot_df.insert(0, "scene", scene_name.name, False)
            annot_df.insert(1, "video", video_name.name, False)

            frames.append(annot_df)

    lookuptable.set_index(["scene", "video", "timestep"], inplace=True)
    lookuptable.sort_index(inplace=True)
    frames = pd.concat(frames)
    frames.set_index(["scene", "video"], inplace=True)
    frames.sort_index(inplace=True)

    # create dataset directory
    print(f"\n\nCreating target directory:\n{save_dir}\n")
    os.mkdir(save_dir)

    # save lookuptable and frames to a pickle
    save_pickle_path = os.path.join(save_dir, f"dataset.pickle")
    print(f"Saving preprocessed dataset to a pickle file:\n{save_pickle_path}\n")
    with open(save_pickle_path, "wb") as f:
        pickle.dump((frames, lookuptable), f)

    # save metadata as a json with same file name as pickle file
    metadata_dict = {
        "orig_fps": original_fps,
        "fps": fps,
        "T_obs": T_obs,
        "T_pred": T_pred,
        "min_n": min_n_agents,
        "agent_classes": agent_classes,
        "other_agents": other_agents
    }
    metadata_path = os.path.join(save_dir, f"dataset_parameters.json")
    print(f"Saving metadata to a json file:\n{metadata_path}\n")
    with open(metadata_path, "w", encoding="utf8") as f:
        json.dump(metadata_dict, f, indent=4)

    print("Done, Goodbye!")
