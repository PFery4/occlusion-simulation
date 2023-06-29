from typing import Tuple
from io import TextIOWrapper

# AgentFormer imports
from utils.config import Config


class AgentFormerDataGenerator:
    """
    This class wraps the dataset classes implemented in this repository in such a way that they are directly usable
    as 'generator' objects in the source code of AgentFormer: https://github.com/Khrylx/AgentFormer
    """
    def __init__(self, parser: Config, log: TextIOWrapper, split: str = 'train', phase: str = 'training'):
        pass

    def shuffle(self) -> None:
        pass

    def get_seq_and_frame(self, index: int) -> Tuple[int, int]:
        pass

    def is_epoch_end(self) -> bool:
        pass

    def next_sample(self) -> dict:
        pass

    def __call__(self) -> dict:
        pass


if __name__ == '__main__':
    print("Hello World!")
