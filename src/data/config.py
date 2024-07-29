import os
import yaml
import pandas as pd

# the directory to the root of the repository
REPO_ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
SDD_CONFIG = 'SDD_dataset_config'   # name of the config file that points to the root of the unprocessed SDD directory.

# homography data: for each scene, we measure a distance in pixel space and real physical space.
# We make use of landmarks present in the scene to obtain corresponding measurements in image and real space.
# Bear in mind that the measurements were taken from one individual image for a whole video.
# The purpose of those measurements is to obtain a *rough* estimate for a pixel/meter conversion.
COORD_CONV = pd.read_csv(
    os.path.join(REPO_ROOT, 'config', 'coordinates_conversion.txt'),
    sep=';', index_col=('scene', 'video')
)

# The splits were defined by random assignment of each scene/video within the SDD.
# The following 5 lines of code can reproduce the split assignment.
# from random import Random
# rng = Random(0)
# SCENE_SPLIT = ["train"] * 42 + ["val"] * 9 + ["test"] * 9
# rng.shuffle(SCENE_SPLIT)
# print(SCENE_SPLIT)
SCENE_SPLIT = pd.read_csv(
    os.path.join(REPO_ROOT, 'config', 'SDD_splits.txt'),
    sep=';', index_col=('scene', 'video')
)


def get_config(config_filename) -> dict:
    # TODO: change input to os.path object instead of str.
    """
    reads the config file '{config_filename}.yaml' that is inside the 'config' directory.
    :return: the contents of the config file, as a dict
    """
    # define the config file path
    confpath = os.path.join(REPO_ROOT, "config", f"{config_filename}.yaml")
    assert os.path.exists(confpath), f"ERROR | PATH DOES NOT EXIST:\n{confpath}"
    # print(f"Loading config from:\n{confpath}\n")

    # read config file
    with open(confpath, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(config)
        except yaml.YAMLError as exc:
            print(exc)
    return config


if __name__ == '__main__':
    print(get_config("config"))
