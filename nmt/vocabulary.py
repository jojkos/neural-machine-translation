from nltk import FreqDist
from nmt import SpecialSymbols
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Vocabulary(object):
    """
        Vocabulary used in Dataset class, handles all the tokens that are used for each language
    """

    def merge_word_seq(self, word_seq):
        for seq in word_seq:
            for word in seq:
                yield word

    def __init__(self, word_seq, max_vocab_size):
        """

        Args:
            word_seq (:obj:`list` of :obj:`str`):
            max_vocab_size: maximum size of the vocabulary, rest will be OOV
        """

        # Creating the vocabulary set with the most common words

        # cannot use keras tokenizer, because we need to add our SpecialSymbols in the vocabuly and keras don't do that
        logger.debug("loading word_seq into FreqDist")
        # dist = FreqDist(np.concatenate(word_seq))
        dist = FreqDist(self.merge_word_seq(word_seq)) # use generator to prevent memory fail
        logger.debug("finding most common {} tokens".format(max_vocab_size))
        vocab = dist.most_common(max_vocab_size)

        logger.debug("Truncating {} different words to {} words".format(len(dist), max_vocab_size))

        # Creating an array of words from the vocabulary set,
        # we will use this array as index-to-word dictionary
        self.ix_to_word = [word[0] for word in vocab]

        # Add PAD at zero index, because zero index is masked in embedding layer
        # has to be in this correct order, to add them all properly on the right spots
        self._insert_symbol_to_vocab(self.ix_to_word, SpecialSymbols.PAD, SpecialSymbols.PAD_IX)
        self._insert_symbol_to_vocab(self.ix_to_word, SpecialSymbols.GO, SpecialSymbols.GO_IX)
        self._insert_symbol_to_vocab(self.ix_to_word, SpecialSymbols.EOS, SpecialSymbols.EOS_IX)
        self._insert_symbol_to_vocab(self.ix_to_word, SpecialSymbols.UNK, SpecialSymbols.UNK_IX)

        self.ix_to_word = {index: word for index, word in enumerate(self.ix_to_word)}
        self.word_to_ix = {self.ix_to_word[ix]: ix for ix in self.ix_to_word}

        # https://github.com/fchollet/keras/issues/6480
        # https://github.com/fchollet/keras/issues/3325
        self.vocab_len = len(self.ix_to_word)

    @staticmethod
    def _insert_symbol_to_vocab(vocab, symbol, index):
        """
        symbol can potentially (for instance as a result of tokenizing where _go and _eos are added to sequence)
        be already part of vocabulary, but we want it to be on specific index
        """

        if symbol in vocab:
            vocab.remove(symbol)

        vocab.insert(index, symbol)

        return vocab

    def get_word(self, ix):
        """

        Args:
            ix (int): index in the vocabulary

        Returns:
             str: word from vocabulary on index ix

        """
        return self.ix_to_word[ix]

    def get_index(self, word):
        """

        Args:
            word (str): word in the vocabulary

        Returns:
             int: index of the word in the vocabulary

        """
        return self.word_to_ix[word]
