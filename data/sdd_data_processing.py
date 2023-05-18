import numpy as np
import pandas as pd
from typing import List


def bool_columns_in(annot_df: pd.DataFrame):
    """
    transforms relevant columns into boolean columns from the annotation dataframe
    :param annot_df: the annotation dataframe
    :return: annot_df
    """
    annot_df["lost"] = annot_df["lost"].astype(bool)
    annot_df["occl."] = annot_df["occl."].astype(bool)
    annot_df["gen."] = annot_df["gen."].astype(bool)
    return annot_df


def xy_columns_in(annot_df: pd.DataFrame):
    """
    adds centroid positions to the annotation dataframe, using bounding box coordinates
    :param annot_df: annotation dataframe
    :return: annot_df
    """
    annot_df["x"] = ((annot_df["xmin"] + annot_df["xmax"]) / 2).astype(int)
    annot_df["y"] = ((annot_df["ymin"] + annot_df["ymax"]) / 2).astype(int)
    return annot_df


def completely_lost_trajs_removed_from(annot_df: pd.DataFrame):
    """
    deletes trajectories from the annotation dataset for which all timesteps are 'lost'
    (i.e., the 'lost' column has value 1)
    :param annot_df: the annotation dataframe of the scene
    :return: annot_df
    """
    agent_ids = annot_df["Id"].unique()
    for agent_id in agent_ids:
        agent_df = annot_df[annot_df["Id"] == agent_id]

        # removing agents from the dataframe if all of their sequence elements are lost
        if np.all(agent_df["lost"].values):
            # print(f"DELETING AGENT {agent_id}: ALL LOST")
            annot_df = annot_df[annot_df["Id"] != agent_id]

        # also deleting trajectories which are way too short to conduct any meaningful kind of prediction
        if np.count_nonzero(~agent_df["lost"].values) < 4:      # TODO: maybe change the '4' to a more meaningful value later (perhaps t_obs + t_pred, something like this...)
            # print(f"DELETING AGENT {agent_id}: TOO SHORT")
            annot_df = annot_df[annot_df["Id"] != agent_id]
    return annot_df


def get_keep_mask_from(agent_df: pd.DataFrame):
    """
    Generates mask values from an agent's trajectory dataframe
    by accounting for the 'lost' frames in the following manner:
    'filter out the lost annotations, and then if this splits the trajectory, keep only the first portion'

    This protocol follows from the remarks mentioned in the paper:
    \"The Stanford Drone Dataset is More Complex than We Think: An Analysis of Key Characteristics\" - Andle et al.
    https://arxiv.org/abs/2203.11743

    :param agent_df: the trajectory datafrome of one individual agent
    :return: the mask of timesteps to keep
    """
    nonlosts = ~agent_df["lost"].values        # array of TRUE / FALSE, showing the opposite value of 'lost'
    keep_mask = np.zeros(len(agent_df.index)).astype(bool)
    keep_idx = np.nonzero(nonlosts)[0][0]     # the index of the first TRUE entry in nonlosts
    while nonlosts[keep_idx]:
        keep_mask[keep_idx] = True
        if keep_idx == len(nonlosts)-1:
            break
        keep_idx += 1
    return keep_mask


def keep_masks_in(annot_df: pd.DataFrame):
    """
    applies get_keep_mask_from() to every individual agent in the dataframe, and adds a new column with the obtained
    masking values.
    :param annot_df:
    :return: annot_df
    """
    annot_df["keep"] = False
    for agent_id in annot_df["Id"].unique():
        keep_mask = get_keep_mask_from(annot_df[annot_df["Id"] == agent_id])
        annot_df.loc[annot_df["Id"] == agent_id, "keep"] = keep_mask
    return annot_df


def subsample_timesteps_from(annot_df: pd.DataFrame, target_fps: float = 2.5, orig_fps: float = 30):
    """
    subsamples the annotation dataframe such that measurements are at the target fps

    the default values for this function follow from the paper:
    \"The Stanford Drone Dataset is More Complex than We Think: An Analysis of Key Characteristics\" - Andle et al.
    https://arxiv.org/abs/2203.11743

    CAREFUL WHEN SAMPLING AT OUT OF PHASE FPS
    (the script does not verify that the resulting target fps will actually be the one desired)

    :param annot_df: the scene annotation dataframe
    :param target_fps: the target fps to sample annotations with
    :param orig_fps: the original annotation fps
    :return: the subsampled annotation dataframe
    """
    t_start = annot_df["frame"].min()
    t_end = annot_df["frame"].max()

    frameskip = int(orig_fps // target_fps)
    target_timesteps = list(range(t_start, t_end, frameskip))

    annot_df = annot_df[annot_df["frame"].isin(target_timesteps)]
    return annot_df


def perform_preprocessing_pipeline(
        annot_df: pd.DataFrame, agent_types: List[str], target_fps: float = 2.5, orig_fps: float = 30
):
    """
    performs the entire following preprocessing pipeline on the provided pandas dataframe:
        - removal of every agent not belonging to the classes specified by agent_types
        - transformation of relevant columns into boolean columns
        - removal of trajectories for which all timesteps are "lost"
        - addition of centroid positions
        - filters out partially "lost" trajectories according to protocol from Andle et.al.
        - subsamples trajectory positions for a target framerate
    :param annot_df: the dataframe to run the preprocessing on
    :param agent_types: the agent classes to keep.
                        must be a subset of ["Pedestrian", "Biker", "Skater", "Car", "Bus", "Cart"]
    :param target_fps: the target fps to sample annotations with
    :param orig_fps: the original annotation fps
    :return: the preprocessed dataframe
    """

    out_df = annot_df[annot_df["label"].isin(agent_types)]
    out_df = bool_columns_in(out_df)
    out_df = completely_lost_trajs_removed_from(out_df)
    out_df = xy_columns_in(out_df)
    out_df = keep_masks_in(out_df)
    out_df = out_df[out_df["keep"]]
    out_df = subsample_timesteps_from(out_df, target_fps=target_fps, orig_fps=orig_fps)

    return out_df
