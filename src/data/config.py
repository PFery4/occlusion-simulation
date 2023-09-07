import os
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))


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
