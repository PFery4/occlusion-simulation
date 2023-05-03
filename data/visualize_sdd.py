import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def get_config() -> dict:
    """
    reads the config file 'config.yaml' at the root of the project repository
    :return: the contents of the config file, as a dict
    """
    # read config file
    confpath = os.path.abspath(os.path.realpath(__file__) + "/../../config.yaml")
    print(f"Loading config from: {confpath}")
    with open(confpath, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(config)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def main():
    print("Hello World!")

    col_names = ["Id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occl.", "gen.", "label"]

    data_path = get_config()["dataset"]["path"]
    print(data_path)

    for scene_name in os.scandir(os.path.join(data_path, "annotations")):
        # print(os.path.realpath(scene_name))
        for video_name in os.scandir(os.path.realpath(scene_name)):
            annot_file_path = os.path.join(data_path, "annotations", scene_name, video_name, "annotations.txt")
            sample_image_path = os.path.join(data_path, "annotations", scene_name, video_name, "reference.jpg")
            # print(annot_file_path)
            # print(os.path.exists(annot_file_path))
            # print(sample_image_path)
            # print(os.path.exists(sample_image_path))

            annot_file_df = pd.read_csv(annot_file_path, sep=" ", names=col_names)

            # we only care about pedestrians:
            is_pedestrian = annot_file_df["label"] == "Pedestrian"
            # we highlight manually annotated points
            not_is_gen = annot_file_df["gen."] == 0
            annot_file_df = annot_file_df[is_pedestrian]
            annot_file_df = annot_file_df[not_is_gen]

            annot_file_df["x"] = (annot_file_df["xmin"] + annot_file_df["xmax"]) / 2
            annot_file_df["y"] = (annot_file_df["ymin"] + annot_file_df["ymax"]) / 2

            agents = annot_file_df["Id"].unique()

            plt.figure(f"{scene_name.name}: {video_name.name}")

            sample_image = cv2.imread(sample_image_path)
            plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

            for agent in agents:
                x = annot_file_df[annot_file_df["Id"] == agent].loc[:, "x"].values
                y = annot_file_df[annot_file_df["Id"] == agent].loc[:, "y"].values
                plt.plot(x, y)
                plt.scatter(x, y, s=1, c='b', marker=',')

        plt.show()


if __name__ == '__main__':
    main()
