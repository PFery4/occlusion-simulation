import os
import yaml
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
# homography data: for each scene, we measure a distance in pixel space and real physical space.
# We make use of landmarks present in the scene to obtain corresponding measurements in image and real space.
# Bear in mind that the measurements were taken from one individual image for a whole scene.
# Scenes contain multiple videos; we assume every video for a given scene is take from roughly the same altitude
# (this might not always be the case, as with deathCircle video3 for example, which is filming closer to the ground
# than the other videos of this scene). The purpose of those measurements is to obtain a *rough* estimate for a
# pixel/meter conversion
PX_PER_M = pd.read_csv(
    os.path.join(REPO_ROOT, "config", "pixel_to_meter.txt"),
    sep=", ", index_col=('scene', 'video'), engine='python'
)


def get_config(config_filename) -> dict:
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
