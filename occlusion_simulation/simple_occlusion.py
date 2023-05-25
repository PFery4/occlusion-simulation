import matplotlib.pyplot as plt
import data.sdd_dataloader as sdd_dataloader
import data.sdd_extract as sdd_extract
import data.sdd_visualize as sdd_visualize

if __name__ == '__main__':
    print("Ok, let's do this")

    instance_idx = 36225

    config = sdd_extract.get_config()
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    instance_dict = dataset.__getitem__(instance_idx)

    fig, ax = plt.subplots()

    sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict)

    plt.show()

