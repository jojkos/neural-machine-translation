import logging

import nmt.utils as utils
import sys
from nmt import SpecialSymbols

logger = logging.getLogger(__name__)


class Dataset(object):
    """
        Class encapsuling loading of the dataset from file
    """

    def __init__(self, dataset_path, source_lang, target_lang, num_samples, tokenize):
        """

        Args:
            dataset_path (str): path to the dataset
            source_lang (str): source language code (extension of the dataset file)
            target_lang (str): target language code (extension of the dataset file)
            num_samples (int): how many lines from the file should be read
            tokenize (bool): whether the words should be tokenized
        """
        self.dataset_path = dataset_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.num_samples = num_samples if num_samples > 0 else None

        self._prepare_dataset(tokenize)

        self.num_samples = len(self.x_word_seq)

        assert len(self.x_word_seq) == len(self.y_word_seq), \
            "dataset %s - has different number of source and target sequences %s %s" % (
                dataset_path, len(self.x_word_seq), len(self.y_word_seq))

    def _prepare_dataset(self, tokenize):
        """

        Loads both dataset files and stores them as sequences. Stores max seq lens as well.

        Args:
            tokenize (bool): whether the words should be tokenized

        """
        x_file_path = "{}.{}".format(self.dataset_path, self.source_lang)
        x_lines = utils.read_file_to_lines(x_file_path, self.num_samples)

        y_file_path = "{}.{}".format(self.dataset_path, self.target_lang)
        y_lines = utils.read_file_to_lines(y_file_path, self.num_samples)

        # it seems that some corpuses (like WMT news commentary) has some lines empty for source or target language

        if tokenize:
            self.x_word_seq = utils.tokenize(x_lines)
            self.y_word_seq = utils.tokenize(y_lines)
        else:
            logger.debug("splitting {} x sequences".format(self.dataset_path))
            # splitting is done in place
            utils.split_lines(x_lines)
            self.x_word_seq = x_lines

            logger.debug("splitting {} y sequences".format(self.dataset_path))
            utils.split_lines(y_lines)
            self.y_word_seq = y_lines

        logger.debug("encapsulating y sequences with special symbols")
        sys.stdout.flush()
        for i in range(len(self.y_word_seq)):
            self.y_word_seq[i] = [SpecialSymbols.GO] + self.y_word_seq[i] + [SpecialSymbols.EOS]

        logger.debug("finding max x seq len")
        self.x_max_seq_len = max(len(seq) for seq in self.x_word_seq)
        logger.debug("finding max y seq len")
        self.y_max_seq_len = max(len(seq) for seq in self.y_word_seq)

        logger.info("Max sequence length for inputs: {}".format(self.x_max_seq_len))
        logger.info("Max sequence length for targets: {}".format(self.y_max_seq_len))

    # in folder code


# python -m nmt.dataset
if __name__ == "__main__":
    dataset = Dataset("data/news-commentary-v9.cs-en",
                      "cs", "en", 100, True)
    print(dataset.x_max_seq_len)
    print(len(dataset.x_word_seq))
