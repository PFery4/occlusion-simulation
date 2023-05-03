import os
import pandas as pd
import yaml

SDD_COL_NAMES = ["Id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occl.", "gen.", "label"]


def get_config() -> dict:
    """
    reads the config file 'config.yaml' at the root of the project repository
    :return: the contents of the config file, as a dict
    """
    # read config file
    confpath = os.path.abspath(os.path.realpath(__file__) + "/../../config.yaml")
    assert os.path.exists(confpath), f"ERROR | PATH DOES NOT EXIST:\n{confpath}"
    # print(f"Loading config from:\n{confpath}\n")

    with open(confpath, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(config)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def pd_df_from(annotation_filepath):
    """
    takes the path of an annotation.txt file of the SDD, and produces its corresponding pandas dataframe
    :param annotation_filepath: the path of the annotation file
    :return: a pandas dataframe
    """
    return pd.read_csv(annotation_filepath, sep=" ", names=SDD_COL_NAMES)


if __name__ == '__main__':
    print(get_config())
