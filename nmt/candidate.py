import logging
from nmt import SpecialSymbols
import numpy as np

logger = logging.getLogger(__name__)


class Candidate(object):
    """
        Temporary candidate or hypotheses of the translated sequence used in beam search
    """

    def __init__(self, target_seq, last_prediction, decoded_sentence, states_value, score, sampled_word=None):
        """

        Args:
            target_seq (numpy array): so far predicted indices of target seq
            last_prediction (int): last prediction index
            decoded_sentence (str): result sequence so far
            states_value: values of inner state and memory cell
            score (int): score so far
            sampled_word (str): last sampled word
        """
        # Generate empty target sequence of length 1.
        self.last_prediction = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        self.last_prediction[0, 0] = last_prediction

        self.target_seq = np.hstack((target_seq, self.last_prediction))

        self.decoded_sentence = decoded_sentence
        self.score = score  # the lower, the better
        self.states_value = states_value
        self.decoded_sentence = decoded_sentence
        self.finalised = False

        if sampled_word == SpecialSymbols.EOS:
            self.finalise()
        elif sampled_word:
            self.decoded_sentence += sampled_word + " "

    def finalise(self):
        """

        Returns: Set the Candidate as finalised - not to be used in another predicting, the sequence is complete.

        """
        self.decoded_sentence = self.decoded_sentence.strip()
        self.finalised = True

    def get_sentence_length(self):
        """
        Returns current length of the candidate sequence

        Returns: current length of the candidate sequence

        """
        return len(self.decoded_sentence.split(" "))
