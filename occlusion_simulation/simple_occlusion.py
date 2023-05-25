import matplotlib.pyplot as plt
import data.sdd_dataloader as sdd_dataloader
import data.sdd_extract as sdd_extract
import data.sdd_visualize as sdd_visualize


def place_ego(instance_dict: dict):

    agent_id = 4

    past = instance_dict["pasts"][agent_id]
    future = instance_dict["futures"][agent_id]
    label = instance_dict["labels"][agent_id]
    full_ons = instance_dict["is_fully_observed"][agent_id]

    for k, v in instance_dict.items():
        print(f"{k}\t\t{v}")


def place_occluder(instance_dict: dict):
    pass


if __name__ == '__main__':
    print("Ok, let's do this")

    instance_idx = 36225

    config = sdd_extract.get_config()
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    instance_dict = dataset.__getitem__(instance_idx)

    fig, ax = plt.subplots()

    sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict)

    place_ego(instance_dict=instance_dict)

    plt.show()

