from typing import Tuple
from io import TextIOWrapper

# AgentFormer imports
from utils.config import Config


class AgentFormerDataGeneratorForSDD:
    """
    This class wraps the dataset classes implemented in this repository in such a way that they are directly usable
    as 'generator' objects in the source code of AgentFormer: https://github.com/Khrylx/AgentFormer
    """
    def __init__(self, parser: Config, log: TextIOWrapper, split: str = 'train', phase: str = 'training'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
            self.init_frame = 0
        elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = parser.data_root_ethucy
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':
            self.sequence_to_load = seq_train
        elif self.split == 'val':
            self.sequence_to_load = seq_val
        elif self.split == 'test':
            self.sequence_to_load = seq_test
        else:
            assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, parser, log, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (
                        parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)

        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

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
