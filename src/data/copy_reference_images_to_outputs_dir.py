import os
import shutil

import src.data.config as conf


if __name__ == '__main__':
    config_dict = conf.get_config(conf.SDD_CONFIG)
    data_path = config_dict["path"]
    output_path = os.path.join(conf.REPO_ROOT, 'outputs', 'figures', 'reference_images')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for scene in os.scandir(os.path.join(data_path, "annotations")):
        for video in os.scandir(os.path.realpath(scene)):
            img_file = os.path.join(os.path.realpath(video), "reference.jpg")
            assert os.path.exists(img_file)
            out_file = os.path.join(output_path, f"{scene.name}_{video.name}.jpg")
            shutil.copy(img_file, out_file)
