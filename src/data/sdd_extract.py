import os
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))


def get_config(config_filename) -> dict:
    """
    reads the config file '{config_filename}.yaml' at the root of the project repository
    :return: the contents of the config file, as a dict
    """
    # read config file
    confpath = os.path.join(REPO_ROOT, "config", f"{config_filename}.yaml")
    assert os.path.exists(confpath), f"ERROR | PATH DOES NOT EXIST:\n{confpath}"
    # print(f"Loading config from:\n{confpath}\n")

    with open(confpath, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            # print(config)
        except yaml.YAMLError as exc:
            print(exc)
    # perform input verifications
    assert config["hyperparameters"]["other_agents"] in ("IN", "OUT")
    return config


if __name__ == '__main__':
    print(get_config("config"))
