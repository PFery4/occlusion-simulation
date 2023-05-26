import matplotlib.pyplot as plt
import numpy as np
import data.sdd_dataloader as sdd_dataloader
import data.sdd_extract as sdd_extract
import data.sdd_visualize as sdd_visualize


def point_between(point_1: np.array, point_2: np.array, k):
    # k between 0 and 1, point_1 and point_2 of same shape
    return k * point_1 + (1 - k) * point_2


def place_ego(instance_dict: dict):

    agent_id = 12
    idx = instance_dict["agent_ids"].index(agent_id)
    past = instance_dict["pasts"][idx]
    future = instance_dict["futures"][idx]
    label = instance_dict["labels"][idx]
    full_ons = instance_dict["is_fully_observed"][idx]

    fig, ax = plt.subplots()
    sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict)

    # pick random past point
    min_obs = 2     # minimum amount of timesteps we want to have observed within observation window
    T_occl = np.random.randint(0, past.shape[0] - min_obs) # number of timesteps to move to the past from t_0 (i.e., 0 -> t_0, 1 -> t_-1, etc)
    past_idx = -1-T_occl

    ax.scatter(past[past_idx][0], past[past_idx][1], marker="x", c="red")

    # pick random future point
    min_reobs = 2   # minimum amount of timesteps we want to be able to reobserve after disocclusion
    T_disoccl = future_idx = np.random.randint(0, future.shape[0] - min_reobs)

    ax.scatter(future[future_idx][0], future[future_idx][1], marker="x", c="orange")

    print(f"{T_occl=}")
    print(f"{T_disoccl=}")

    p_occl = point_between(past[past_idx], past[past_idx-1], np.random.random())
    p_disoccl = point_between(future[future_idx], future[future_idx+1], np.random.random())

    ax.scatter(p_occl[0], p_occl[1], marker="x", c="yellow")
    ax.scatter(p_disoccl[0], p_disoccl[1], marker="x", c="green")

    midpoint = point_between(p_occl, p_disoccl, 0.5)

    ax.scatter(midpoint[0], midpoint[1], marker="x", c="blue")



    # for i in range(100):
    #     print(random_point_between(past, future))

    plt.show()



def place_occluder(instance_dict: dict):
    pass


if __name__ == '__main__':
    print("Ok, let's do this")

    instance_idx = 36225

    config = sdd_extract.get_config()
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    instance_dict = dataset.__getitem__(instance_idx)

    place_ego(instance_dict=instance_dict)


