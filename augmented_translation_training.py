import os
from time import sleep
from threading import Thread
from fairseq.tasks.translation import TranslationTask, register_task
from fairseq import utils


FAIRSEQ_DICT_FORMAT = 'dict.{lang}.txt'
SLEEP_INTERVAL = 5  # while waiting for data processing to finish


@register_task('augmented-translation')
class AugmentedTranslation(TranslationTask):
    """Fairseq task for training a translation model while preprocessing the data each epoch.
    The motivation is to train using different augmentation and regularization techniques,
    such as subword sampling and BPE-dropout.
    This is only a template, though. You'll probably need to add your arguments in add_args method,
    and you'll definitely need to write your preprocessing code in _preprocess method."""

    @staticmethod
    def add_args(parser):
        # Add your arguments here
        parser.add_argument('--original-data', required=True,
                            help='Path to un-processed training data')
        super(AugmentedTranslation, AugmentedTranslation).add_args(parser)

    @classmethod
    def setup_task(cls, args, **kwargs):
        if not getattr(args, 'source_lang', None) or not getattr(args, 'target_lang', None):
            raise ValueError('Please specify --source-lang and --target-lang')

        data_dirs = utils.split_paths(args.data)
        if len(data_dirs) != 2 or data_dirs[0] == data_dirs[1]:
            raise ValueError('This task requires two data directories')

        if not os.path.exists(data_dirs[1]):
            os.mkdir(data_dirs[1])

        cls._preprocess(args, data_dirs[0], first_epoch=True)

        return super().setup_task(args, **kwargs)

    def __init__(self, args, src_dict, tgt_dict):
        self.next_dataset_thread = None
        self.current_directory, self.next_directory = utils.split_paths(args.data)
        super().__init__(args, src_dict, tgt_dict)

    @staticmethod
    def _preprocess(args, data_path, first_epoch=False):
        # Use same dictionaries
        first_data_dir = utils.split_paths(args.data)[0]
        src_dict = os.path.join(first_data_dir, FAIRSEQ_DICT_FORMAT.format(lang=args.source_lang))
        tgt_dict = os.path.join(first_data_dir, FAIRSEQ_DICT_FORMAT.format(lang=args.target_lang))

        # Your preprocessing code goes here

    def _wait_for_preprocessing(self):
        if self.next_dataset_thread is not None:
            sleep_total = 0
            while self.next_dataset_thread.is_alive():
                if sleep_total % 100 == 0:
                    print(f'Waiting for data preprocessing (slept for {sleep_total} seconds so far)')
                sleep(SLEEP_INTERVAL)
                sleep_total += SLEEP_INTERVAL
            self.next_dataset_thread.join()

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self._wait_for_preprocessing()
        super().load_dataset(split, epoch, combine, **kwargs)
        if split == 'train':
            self.next_dataset_thread = Thread(target=self._preprocess_next_dataset)
            self.next_dataset_thread.start()

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
    ):
        for subset in self.args.valid_subset.split(','):
            if dataset == self.dataset(subset):
                self._wait_for_preprocessing()
        return super().get_batch_iterator(dataset, max_tokens, max_sentences, max_positions, ignore_invalid_inputs,
                                          required_batch_size_multiple, seed, num_shards, shard_id, num_workers, epoch)

    def _preprocess_next_dataset(self):
        self._preprocess(self.args, data_path=self.next_directory)
        self.current_directory, self.next_directory = self.next_directory, self.current_directory
