import os
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
# homography data: for each scene, we measure a distance in pixel space and real physical space.
# We make use of landmarks present in the scene to obtain corresponding measurements in image and real space.
# Bear in mind that the measurements were taken from one individual image for a whole scene.
# Scenes contain multiple videos; we assume every video for a given scene is take from roughly the same altitude
# (this might not always be the case, as with deathCircle video3 for example, which is filming closer to the ground
# than the other videos of this scene). The purpose of those measurements is to obtain a *rough* estimate for a
# pixel/meter conversion
# HOMOGRAPHY_DATA = {             # [px, m]
#     'bookstore': [668.4, 25.3],
#     'coupa': [814.3, 22.7],
#     'deathCircle': [422.1, 17.47],
#     'gates': [544.3, 20.88],
#     'hyang': [933.0, 31.73],
#     'little': [677.4, 19.81],
#     'nexus': [783.0, 34.03],
#     'quad': [820.1, 35.94],
# }
PX_PER_M = 27.74344283640459        # [px/m], the average px/m value across every scene in HOMOGRAPHY_DATA


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
