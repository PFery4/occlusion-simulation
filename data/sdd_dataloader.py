import os.path

import torch
from torch.utils.data import Dataset
import sdd_extract
import sdd_data_processing


class StanfordDroneDataset(Dataset):

    def __init__(self, config):
        self.root = config["dataset"]["path"]

        # read the csv's for all videos
        # hopefully it is manageable within memory
        for scene_name in os.scandir(os.path.join(self.root, "annotations")):
            for video_name in os.scandir(os.path.realpath(scene_name)):
                annot_file_path = os.path.join(os.path.realpath(video_name), "annotations.txt")
                print(annot_file_path)
                assert os.path.exists(annot_file_path)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def all_subfolders(self):
        print(os.path.join(self.root, "annotations"))

        for scene_name in os.scandir(os.path.join(self.root, "annotations")):
            print(scene_name.path)
            print(sorted(os.listdir(scene_name.path)))
        return sorted(os.listdir(os.path.join(self.root, "annotations")))


if __name__ == '__main__':

    config = sdd_extract.get_config()

    dataset = StanfordDroneDataset(config=config)
    print(dataset.all_subfolders())
