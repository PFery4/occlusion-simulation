import sys
import numpy as np
import pandas as pd


def make_bool_columns(annot_df: pd.DataFrame):
    """
    transforms relevant columns into boolean columns from the annotation dataframe
    :param annot_df: the annotation dataframe
    :return: annot_df
    """
    annot_df["lost"] = annot_df["lost"].astype(bool)
    annot_df["occl."] = annot_df["occl."].astype(bool)
    annot_df["gen."] = annot_df["gen."].astype(bool)
    return annot_df


def add_xy_columns_to(annot_df: pd.DataFrame):
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
    :return: None
    """
    agent_ids = annot_df["Id"].unique()
    for agent_id in agent_ids:
        agent_df = annot_df[annot_df["Id"] == agent_id]
        # print(np.all(agent_df["lost"].values))
        if np.all(agent_df["lost"].values):
            print(f"DELETING AGENT {agent_id}: ALL LOST")
            annot_df = annot_df[annot_df["Id"] != agent_id]
    return annot_df


def get_keep_mask_from(agent_df: pd.DataFrame):
    """
    Adds a column of mask values to the trajectory dataframe
    by accounting for the 'lost' frames in the following manner:
    'filter out the lost annotations, and then if this splits the trajectory, keep only the first portion'

    This protocol is taken from the remarks mentioned in the paper:
    \"The Stanford Drone Dataset is More Complex than We Think: An Analysis of Key Characteristics\" - Andle et al.
    https://arxiv.org/abs/2203.11743

    :param agent_df: the trajectory datafrome of one individual agent
    :return: the mask of timesteps to keep
    """
    np.set_printoptions(threshold=sys.maxsize)

    losts = agent_df["lost"].values

    # first non-lost timestep
    first_keep_idx = np.where(losts == 0)[0][0]
    # print(losts)
    # print(sum(losts))
    # print(first_keep_idx)
