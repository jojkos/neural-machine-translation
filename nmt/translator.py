# coding: utf-8

import logging
import os
import math
import random

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import numpy as np
import nmt.utils as utils
from nmt import SpecialSymbols, Dataset, Vocabulary, Candidate
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Embedding, Input, Bidirectional, Concatenate, Average, Dropout
from keras.layers.merge import add
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Translator(object):
    """

    Main class of the module, takes care of the datasets, fitting, evaluation and translating

    """

    def __init__(self, source_lang, model_file, model_folder,
                 target_lang, test_dataset, training_dataset,
                 reverse_input=True, max_source_vocab_size=10000, max_target_vocab_size=10000,
                 source_embedding_path=None, target_embedding_path=None,
                 clear=False, tokenize=True, log_folder="logs/", num_units=256, num_threads=1, dropout=0.2,
                 optimizer="rmsprop",
                 source_embedding_dim=300, target_embedding_dim=300,
                 max_source_embedding_num=None, max_target_embedding_num=None,
                 num_training_samples=-1, num_test_samples=-1,
                 num_encoder_layers=1, num_decoder_layers=1):
        """

        Args:
            source_embedding_dim (int): Dimension of embeddings
            target_embedding_dim (int): Dimension of embeddings
            source_embedding_path (str): Path to pretrained fastText embeddings file
            target_embedding_path (str): Path to pretrained fastText embeddings file
            max_source_embedding_num (int): how many first lines from embedding file should be loaded, None means all of them
            max_target_embedding_num (int): how many first lines from embedding file should be loaded, None means all of them
            source_lang (str): Source language (dataset file extension)
            num_units (int): Size of each network layer
            dropout (float): Size of dropout
            optimizer (str): Keras optimizer name
            log_folder (str): Path where the result logs will be stored
            max_source_vocab_size (int): Maximum size of source vocabulary
            max_target_vocab_size (int): Maximum size of target vocabulary
            model_file (str): Model file name. Either will be created or loaded.
            model_folder (str): Path where the result model will be stored
            num_training_samples (int, optional): How many samples to take from the training dataset, -1 for all of them (default)
            num_test_samples (int, optional): How many samples to take from the test dataset, -1 for all of them (default)
            reverse_input (bool): Whether to reverse source sequences (optimization for better learning)
            target_lang (str): Target language (dataset file extension)
            test_dataset (str): Path to the test set. Dataset are two files (one source one target language)
            training_dataset (str): Path to the training set
            clear (bool): Whether to delete old weights and logs before running
            tokenize (bool): Whether to tokenize the sequences or not (they are already tokenizes e.g. using Moses tokenizer)
            num_encoder_layers (int): Number of layers in encoder
            num_decoder_layers (int): Number of layers in decoder
        """

        self.source_embedding_dim = source_embedding_dim
        self.target_embedding_dim = target_embedding_dim
        self.source_embedding_path = source_embedding_path
        self.target_embedding_path = target_embedding_path
        self.max_source_embedding_num = max_source_embedding_num
        self.max_target_embedding_num = max_target_embedding_num
        self.source_lang = source_lang
        self.num_units = num_units
        self.num_threads = num_threads
        self.dropout = dropout
        self.optimizer = optimizer
        self.log_folder = log_folder
        self.max_source_vocab_size = max_source_vocab_size
        self.max_target_vocab_size = max_target_vocab_size
        self.model_folder = model_folder
        self.model_weights_path = "{}".format(os.path.join(model_folder, model_file))
        self.num_training_samples = num_training_samples
        self.num_test_samples = num_test_samples
        self.reverse_input = reverse_input
        self.target_lang = target_lang
        self.test_dataset_path = test_dataset
        self.training_dataset_path = training_dataset
        self.clear = clear
        self.tokenize = tokenize
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        # configure number of threads

        # intra_op_parallelism_threads = self.num_threads,
        # inter_op_parallelism_threads = self.num_threads,
        # allow_soft_placement = True,
        # device_count = {'CPU': self.num_threads}

        # FAILS ON FIT CLUSTER when started manually and not through qsub
        config = tf.ConfigProto(intra_op_parallelism_threads=self.num_threads,
                                inter_op_parallelism_threads=self.num_threads,
                                allow_soft_placement=True,
                                device_count={'CPU': self.num_threads})
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        utils.prepare_folders([self.log_folder, self.model_folder], clear)

        self.training_dataset = Dataset(self.training_dataset_path, self.source_lang, self.target_lang,
                                        self.num_training_samples,
                                        self.tokenize)
        self.test_dataset = Dataset(self.test_dataset_path, self.source_lang, self.target_lang,
                                    self.num_test_samples,
                                    self.tokenize)

        logger.info("There are {} samples in training dataset".format(self.training_dataset.num_samples))
        logger.info("There are {} samples in test dataset".format(self.test_dataset.num_samples))

        self.source_vocab = Vocabulary(self.training_dataset.x_word_seq, self.max_source_vocab_size)
        self.target_vocab = Vocabulary(self.training_dataset.y_word_seq, self.max_target_vocab_size)

        logger.info("Source vocabulary has {} symbols".format(self.source_vocab.vocab_len))
        logger.info("Target vocabulary has {} symbols".format(self.target_vocab.vocab_len))

        self.source_embedding_weights = None
        if not os.path.isfile(self.model_weights_path) and self.source_embedding_path:
            # load pretrained embeddings
            self.source_embedding_weights = utils.load_embedding_weights(self.source_embedding_path,
                                                                         self.source_vocab.ix_to_word,
                                                                         limit=self.max_source_embedding_num)

        self.target_embedding_weights = None
        if not os.path.isfile(self.model_weights_path) and self.target_embedding_path:
            # load pretrained embeddings
            self.target_embedding_weights = utils.load_embedding_weights(self.target_embedding_path,
                                                                         self.target_vocab.ix_to_word,
                                                                         limit=self.max_target_embedding_num)

        self.model, self.encoder_model, self.decoder_model = self._define_models()

        logger.info("Global model")
        self.model.summary()

        logger.info("Encoder model")
        self.encoder_model.summary()

        logger.info("Decoder model")
        self.decoder_model.summary()

        # model_to_dot(self.model).write_pdf("model.pdf")
        # model_to_dot(self.encoder_model).write_pdf("encoder_model.pdf")
        # model_to_dot(self.decoder_model).write_pdf("decoder_model.pdf")

        logger.info("compiling model...")
        # Run training
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                           metrics=['acc'])

        if os.path.isfile(self.model_weights_path):
            logger.info("Loading model weights from file..")
            self.model.load_weights(self.model_weights_path)

    def _get_encoded_data(self, dataset, from_index=0, to_index=None):

        encoder_input_data, decoder_input_data, decoder_target_data = Translator.encode_sequences(
            dataset.x_word_seq[from_index: to_index],
            dataset.y_word_seq[from_index: to_index],
            dataset.x_max_seq_len, dataset.y_max_seq_len,
            self.source_vocab, self.target_vocab, self.reverse_input
        )

        return {
            "encoder_input_data": encoder_input_data,
            "decoder_input_data": decoder_input_data,
            "decoder_target_data": decoder_target_data
        }

    def _get_training_data(self, from_index=0, to_index=None):
        """

        Returns: dict with encoder_input_data, decoder_input_data and decoder_target_data of whole dataset size

        """

        return self._get_encoded_data(self.training_dataset, from_index, to_index)

    def _training_data_gen_WRONG_SHUFFLE(self, batch_size, infinite=True, shuffle=True, bucketing=False,
                                         bucket_range=3):
        """
        Creates generator for keras fit_generator. First yielded value is number of steps needed for whole epoch.

        Args:
            infinite: whether to yield data infinitely or stop after one walkthrough the dataset
            shuffle: whether to shuffle the training data and return them in random order every epoch
            bucketing: whetether to use bucketing
            bucket_range: range of each bucket

        Returns: First yielded value is number of steps needed for whole epoch.
            Then yields ([encoder_input_data, decoder_input_data], decoder_target_data)

        """
        # shuffling
        # https://stackoverflow.com/questions/46570172/how-to-fit-generator-in-keras
        # https://github.com/keras-team/keras/issues/2389

        # first value returned from generator is the number of steps for the whole epoch
        first = True

        if bucketing:
            buckets = utils.split_to_buckets(self.training_dataset.x_word_seq,
                                             self.training_dataset.y_word_seq,
                                             bucket_range,
                                             self.training_dataset.x_max_seq_len,
                                             self.training_dataset.y_max_seq_len,
                                             batch_size)

        while True:
            if bucketing:
                indices = []

                for bucket in sorted(buckets.keys()):
                    bucket_indices = list(range(0, len(buckets[bucket]["x_word_seq"]), batch_size))
                    for index in bucket_indices:
                        indices.append([bucket, index])

                if first:
                    yield len(indices)
                    first = False

                if shuffle:
                    random.shuffle(indices)

                for bucket_ix, i in indices:
                    training_data = Translator.encode_sequences(
                        buckets[bucket_ix]["x_word_seq"][i: i + batch_size],
                        buckets[bucket_ix]["y_word_seq"][i: i + batch_size],
                        buckets[bucket_ix]["x_max_seq_len"],
                        buckets[bucket_ix]["y_max_seq_len"],
                        self.source_vocab, self.target_vocab, self.reverse_input
                    )
                    yield [training_data[0], training_data[1]], training_data[2]
            else:
                indices = list(range(0, self.training_dataset.num_samples, batch_size))

                if first:
                    yield len(indices)
                    first = False

                if shuffle:
                    random.shuffle(indices)

                for i in indices:
                    training_data = self._get_training_data(i, i + batch_size)
                    yield [training_data["encoder_input_data"], training_data["decoder_input_data"]], training_data[
                        "decoder_target_data"]

            if not infinite:
                break

    def _training_data_gen(self, batch_size, infinite=True, shuffle=True):
        """
        Creates generator for keras fit_generator. First yielded value is number of steps needed for whole epoch.

        Args:
            infinite: whether to yield data infinitely or stop after one walkthrough the dataset
            shuffle: whether to shuffle the training data and return them in random order every epoch

        Returns: First yielded value is number of steps needed for whole epoch.
            Then yields ([encoder_input_data, decoder_input_data], decoder_target_data)

        """
        # shuffling
        # https://stackoverflow.com/questions/46570172/how-to-fit-generator-in-keras
        # https://github.com/keras-team/keras/issues/2389

        # first value returned from generator is the number of steps for the whole epoch
        first = True

        while True:
            x = []
            y = []
            indices = list(range(0, self.training_dataset.num_samples))

            if first:
                yield math.ceil(len(indices) / batch_size)
                first = False

            if shuffle:
                random.shuffle(indices)

            for ix in indices:
                x.append(self.training_dataset.x_word_seq[ix])
                y.append(self.training_dataset.y_word_seq[ix])

            i = 0

            while i < len(indices):
                encoder_input_data, decoder_input_data, decoder_target_data = Translator.encode_sequences(
                    x[i: i + batch_size],
                    y[i: i + batch_size],
                    self.training_dataset.x_max_seq_len, self.training_dataset.y_max_seq_len,
                    self.source_vocab, self.target_vocab, self.reverse_input
                )

                yield [encoder_input_data, decoder_input_data], decoder_target_data

                i += batch_size

            if not infinite:
                break

    def _training_data_bucketing(self, batch_size, infinite=True, shuffle=True, bucket_range=3):
        """
        Creates generator for keras fit_generator. First yielded value is number of steps needed for whole epoch.

        Args:
            infinite: whether to yield data infinitely or stop after one walkthrough the dataset
            shuffle: whether to shuffle the training data and return them in random order every epoch
            bucket_range: range of each bucket

        Returns: First yielded value is number of steps needed for whole epoch.
            Then yields ([encoder_input_data, decoder_input_data], decoder_target_data)

        """
        # shuffling
        # https://stackoverflow.com/questions/46570172/how-to-fit-generator-in-keras
        # https://github.com/keras-team/keras/issues/2389

        # first value returned from generator is the number of steps for the whole epoch
        first = True

        buckets = utils.split_to_buckets(self.training_dataset.x_word_seq,
                                         self.training_dataset.y_word_seq,
                                         bucket_range,
                                         self.training_dataset.x_max_seq_len,
                                         self.training_dataset.y_max_seq_len,
                                         batch_size)
        indices = []

        # create indices to access each bucket and then each batch inside that bucket
        for bucket in sorted(buckets.keys()):
            bucket_indices = list(range(0, len(buckets[bucket]["x_word_seq"]), batch_size))
            for index in bucket_indices:
                indices.append([bucket, index])

        while True:
            if first:
                yield len(indices)
                first = False

            if shuffle:
                # we need as much random shufflin as possible
                # so we shuffle both data inside buckets and then the order in which they are accessed

                # shuffle all data inside the buckets
                for bucket in sorted(buckets.keys()):
                    zipped = list(zip(buckets[bucket]["x_word_seq"], buckets[bucket]["y_word_seq"]))
                    random.shuffle(zipped)
                    buckets[bucket]["x_word_seq"], buckets[bucket]["y_word_seq"] = zip(*zipped)

                # shuffle the global bucket->batch indices
                random.shuffle(indices)

            for bucket_ix, i in indices:
                training_data = Translator.encode_sequences(
                    buckets[bucket_ix]["x_word_seq"][i: i + batch_size],
                    buckets[bucket_ix]["y_word_seq"][i: i + batch_size],
                    buckets[bucket_ix]["x_max_seq_len"],
                    buckets[bucket_ix]["y_max_seq_len"],
                    self.source_vocab, self.target_vocab, self.reverse_input
                )
                yield [training_data[0], training_data[1]], training_data[2]

            if not infinite:
                break

    def _get_test_data(self, from_index=0, to_index=None):
        return self._get_encoded_data(self.test_dataset, from_index, to_index)

    def _test_data_gen(self, batch_size, infinite=True, return_steps=False):
        """
        # vocabularies of test dataset has to be the same as of training set
        # otherwise embeddings would not correspond are use OOV
        # and y one hot encodings wouldnt correspond either

        Args:
            batch_size (int): size of the batch
            infinite (bool): whether to run infinitely or just do one loop over the dataset
            return_steps (bool): whether to return number of steps for the whole epoch as a first yield

        Yields: x inputs, y inputs

        """

        i = 0
        once_through = False

        if return_steps:
            steps = self.test_dataset.num_samples / batch_size
            yield math.ceil(steps)

        while infinite or not once_through:
            test_data = self._get_test_data(i, i + batch_size)

            yield (
                [test_data["encoder_input_data"], test_data["decoder_input_data"]],
                test_data["decoder_target_data"]
            )

            i += batch_size

            if i >= self.test_dataset.num_samples:
                once_through = True
                i = 0

    @staticmethod
    def encode_sequences(x_word_seq, y_word_seq, x_max_seq_len, y_max_seq_len,
                         source_vocab, target_vocab, reverse_input=True):
        """

        Take word sequences and convert them so that the model can be fit with them.
        Input words are just converted to integer index
        Target words are encoded to one hot vectors of target vocabulary length

        Args:
            x_word_seq: input word sequences
            y_word_seq: target word sequences
            x_max_seq_len (int): max lengh of input word sequences
            y_max_seq_len (int): max lengh of target word sequences
            source_vocab (Vocabulary): source vocabulary object
            target_vocab (Vocabulary): target vocabulary object
            reverse_input (bool): whether to reverse input sequences

        Returns: encoder_input_data, decoder_input_data, decoder_target_data

        """

        # if we try to allocate memory for whole dataset (even for not a big one), Memory Error is raised
        # always encode only a part of the dataset
        encoder_input_data = np.zeros(
            (len(x_word_seq), x_max_seq_len), dtype='float32')
        decoder_input_data = np.zeros(
            (len(x_word_seq), y_max_seq_len - 1), dtype='float32')
        # - 1 because decder_input doesn't take last EOS and decoder_target doesn't take first GO symbol
        decoder_target_data = np.zeros(
            (len(x_word_seq), y_max_seq_len - 1, target_vocab.vocab_len),
            dtype='float32')

        # prepare source sentences for embedding layer (encode to indexes)
        for i, seq in enumerate(x_word_seq):
            if reverse_input:  # for better results according to paper Sequence to seq...
                seq = seq[::-1]
            for t, word in enumerate(seq):
                if word in source_vocab.word_to_ix:
                    encoder_input_data[i, t] = source_vocab.word_to_ix[word]
                else:
                    encoder_input_data[i, t] = SpecialSymbols.UNK_IX

        # encode target sentences to one hot encoding
        for i, seq in enumerate(y_word_seq):
            for t, word in enumerate(seq):
                if word in target_vocab.word_to_ix:
                    index = target_vocab.word_to_ix[word]
                else:
                    index = SpecialSymbols.UNK_IX
                # decoder_target_data is ahead of decoder_input_data by one timestep
                # ignore EOS symbol at the end
                if t < len(seq) - 1:
                    decoder_input_data[i, t] = index

                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, index] = 1

        return encoder_input_data, decoder_input_data, decoder_target_data

    def _define_models(self):
        """

        Defines main model for learning, encoder_model for prediction of encoder state in inference time
        and decoder_model for predicting of results in inference time

        Returns: model, encoder_model, decoder_model

        """
        # model based on https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
        logger.info("Creating models...")
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,), name="encoder_input")

        if self.source_embedding_weights is not None:
            self.source_embedding_weights = [self.source_embedding_weights]  # Embedding layer wantes list as parameter

        # according to https://keras.io/layers/embeddings/
        # input dim should be +1 when used with mask_zero..is it correctly set here?
        # i think that input dim is already +1 because padding symbol is part of the vocabulary
        source_embeddings = Embedding(self.source_vocab.vocab_len, self.source_embedding_dim,
                                      weights=self.source_embedding_weights, mask_zero=True, trainable=True,
                                      name="input_embeddings")
        source_embedding_outputs = source_embeddings(encoder_inputs)

        # use bi-directional encoder with concatenation as in Google neural machine translation paper
        # https://stackoverflow.com/questions/47923370/keras-bidirectional-lstm-seq2seq

        # only first layer is bidirectional (too much params if all of them were)
        # its OK to have return_sequences here as encoder outputs are not used anyway in the decoder and it is needed for multi layer encoder
        bidirectional_encoder = Bidirectional(LSTM(self.num_units, return_state=True, return_sequences=True,
                                                   dropout=self.dropout, recurrent_dropout=self.dropout),
                                              name="bidirectional_encoder_layer")
        # h is inner(output) state, c i memory cell
        inputs = source_embedding_outputs
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = bidirectional_encoder(inputs)
        state_h = Average()([forward_h, backward_h])
        state_c = Average()([forward_c, backward_c])

        # multiple encoder layers
        for i in range(1, self.num_encoder_layers):
            # residual connections as in google's paper https://arxiv.org/abs/1609.08144
            # if inputs (embeddings) size is different than LSTM's, sum has can be performed only from layer 2 on
            # first layer is bidirectional (twice as big output) so it has to be skipped as well (i > 2)
            if i > 2 or inputs.shape[-1] == self.num_units:
                inputs = add([inputs, encoder_outputs])
            else:
                inputs = encoder_outputs

            encoder_outputs, state_h, state_c = LSTM(self.num_units, return_state=True, return_sequences=True,
                                                     dropout=self.dropout, recurrent_dropout=self.dropout,
                                                     name="encoder_layer_{}".format(i + 1))(inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,), name="decoder_input")

        if self.target_embedding_weights is not None:
            self.target_embedding_weights = [self.target_embedding_weights]  # Embedding layer wantes list as parameter
        target_embeddings = Embedding(self.target_vocab.vocab_len, self.target_embedding_dim,
                                      weights=self.target_embedding_weights, mask_zero=True, trainable=True,
                                      name="target_embeddings")
        target_embedding_outputs = target_embeddings(decoder_inputs)

        inputs = target_embedding_outputs
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.num_units, return_sequences=True, return_state=True,
                            dropout=self.dropout, recurrent_dropout=self.dropout,
                            name="decoder_layer_1")
        decoder_outputs, _, _ = decoder_lstm(inputs,
                                             initial_state=encoder_states)

        # multiple decoder layers
        decoder_layers = []
        for i in range(1, self.num_decoder_layers):
            # residual connections
            # if inputs (embeddings) size is different than LSTM's, sum has can be performed only from layer 2 on
            if i > 1 or inputs.shape[-1] == self.num_units:
                inputs = add([inputs, decoder_outputs])
            else:
                inputs = decoder_outputs

            decoder_layers.append(LSTM(self.num_units, return_state=True, return_sequences=True,
                                       name="decoder_layer_{}".format(i + 1)))
            # in the learning model, initial state of all decoder layers is encoder_states
            decoder_outputs, _, _ = decoder_layers[-1](inputs, initial_state=encoder_states)

        decoder_dense = Dense(self.target_vocab.vocab_len, activation='softmax',
                              name="output_layer")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(encoder_inputs, encoder_states)

        inputs = target_embedding_outputs
        decoder_state_input_h = Input(shape=(self.num_units,), name="decoder_state_h_input")
        decoder_state_input_c = Input(shape=(self.num_units,), name="decoder_state_c_input")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, _, _ = decoder_lstm(
            inputs, initial_state=decoder_states_inputs)

        for i, decoder_layer in enumerate(decoder_layers):
            # residual connections
            if i > 0 or inputs.shape[-1] == self.num_units:
                inputs = add([inputs, decoder_outputs])
            else:
                inputs = decoder_outputs

            # every layer has to have its own inputs and outputs, because each outputs different state after first token
            # at the start all of the layers are initialized with encoder states
            # in inference, whole sequence has to be used as an input (not one word after another)
            # to get propper inner states in hidden layers
            decoder_outputs, _, _ = decoder_layer(inputs,
                                                  initial_state=decoder_states_inputs)

        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            decoder_outputs)

        return model, encoder_model, decoder_model

    @staticmethod
    def decode_encoded_seq(seq, vocab, one_hot=False):
        decoded = []
        for ix in seq:
            if one_hot:
                ix = np.argmax(ix)
            decoded.append(vocab.ix_to_word[ix])

        return decoded

    def translate_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = SpecialSymbols.GO_IX

        # Sampling loop for a batch of sequences
        decoded_sentence = ""
        while True:
            outputs = self.decoder_model.predict(
                [target_seq] + states_value)

            # outputs result for each token in sequence, we want to sample the last one
            output_tokens = outputs[0][-1]

            # Sample a token
            sampled_token_index = np.argmax(output_tokens)
            sampled_word = self.target_vocab.ix_to_word[sampled_token_index]

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_word == SpecialSymbols.EOS:
                break

            decoded_sentence += sampled_word + " "
            decoded_len = len(decoded_sentence.strip().split(" "))

            if decoded_len > self.training_dataset.y_max_seq_len \
                    and decoded_len > self.test_dataset.y_max_seq_len:
                break

            # Update the target sequence, add last samples word.
            new_target_token = np.zeros((1, 1))
            new_target_token[0, 0] = sampled_token_index

            target_seq = np.hstack((target_seq, new_target_token))

        # for BPE encoded
        # decoded_sentence = re.sub(r"(@@ )|(@@ ?$)", "", decoded_sentence)

        return decoded_sentence.strip()

    def translate_sequence_beam(self, input_seq, beam_size=1):
        # https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # only one candidate at the begining
        candidates = [
            Candidate(target_seq=target_seq, last_prediction=SpecialSymbols.GO_IX, states_value=states_value, score=0,
                      decoded_sentence="")
        ]

        while True:
            should_stop = True
            new_candidates = []
            for candidate in candidates:
                if not candidate.finalised:
                    outputs = self.decoder_model.predict(
                        [candidate.target_seq] + candidate.states_value)
                    should_stop = False

                    output_tokens = outputs[0][-1]

                    # find n (beam_size) best predictions
                    indices = np.argpartition(output_tokens, -beam_size)[-beam_size:]

                    for sampled_token_index in indices:
                        score = -math.log(output_tokens[sampled_token_index])
                        # how long is the sentence, to compute average score
                        step = candidate.get_sentence_length() + 1

                        # i believe scores should be summed together because log prob is used https://stats.stackexchange.com/questions/121257/log-probability-vs-product-of-probabilities
                        # score is average of all probabilities (normalization so that longer sequences are not penalized)
                        # incremental average https://math.stackexchange.com/questions/106700/incremental-averageing
                        avg_score = utils.incremental_average(candidate.score, score, step)

                        sampled_word = self.target_vocab.ix_to_word[sampled_token_index]

                        new_candidate = Candidate(target_seq=candidate.target_seq,
                                                  states_value=states_value,
                                                  decoded_sentence=candidate.decoded_sentence,
                                                  score=avg_score,
                                                  sampled_word=sampled_word, last_prediction=sampled_token_index)
                        new_candidates.append(new_candidate)

                        # Exit condition: either hit max length
                        # or find stop character.
                        if sampled_word == SpecialSymbols.EOS:
                            continue

                        decoded_len = new_candidate.get_sentence_length()

                        if decoded_len > self.training_dataset.y_max_seq_len \
                                and decoded_len > self.test_dataset.y_max_seq_len:
                            new_candidate.finalise()
                            continue

                # finished candidates are transfered to new_candidates automatically
                else:
                    new_candidates.append(candidate)

            # take n (beam_size) best candidates
            candidates = sorted(new_candidates, key=lambda can: can.score)[:beam_size]

            if should_stop:
                break

        return candidates[0].decoded_sentence

    @staticmethod
    def encode_text_seq_to_encoder_seq(text, vocab):
        """
        Encodes given text sequence to numpy array ready to be used as encoder_input for prediction

        Args:
            text (str): sequence to translate
            vocab (Vocabulary): vocabulary object

        Returns: encoded sequence ready to be used as encoder_inpout for prediction

        """
        sequences = text_to_word_sequence(text)
        x = np.zeros((1, len(sequences)), dtype='float32')

        for i, seq in enumerate(sequences):
            if seq in vocab.word_to_ix:
                ix = vocab.word_to_ix[seq]
            else:
                ix = SpecialSymbols.UNK_IX
            x[0][i] = ix

        return x

    @staticmethod
    def get_gen_steps(dataset, batch_size):
        """

        Returns how many steps are needed for the generator to go through the whole dataset with the batch_size

        Args:
            dataset: dataset that is beeing proccessed
            batch_size: size of the batch

        Returns: number of steps for the generatorto go through whole dataset

        """

        assert batch_size > 0

        return math.ceil(dataset.num_samples / batch_size)

    def fit(self, epochs=1, initial_epoch=0, batch_size=64, use_fit_generator=False,
            bucketing=False, bucket_range=3, early_stopping_patience=-1):
        """

        fits the model, according to the parameters passed in constructor

        Args:
            epochs: Number of epochs
            initial_epoch: Epoch number from which to start
            batch_size: Size of one batch
            use_fit_generator: Prevent memory crash by only load part of the dataset at once each time when fitting
            bucketing (bool): Whether to bucket sequences according their size to optimize padding
                automatically switches use_fit_generator to True
            bucket_range (int): Range of different sequence lenghts in one bucket
            early_stopping_patience (int): How many epochs should model train after loss/val_loss decreases. -1 means that early stopping won't be used.

        """

        if bucketing:
            use_fit_generator = True

        # logging for tensorboard
        tensorboard_callback = TensorBoard(log_dir="{}".format(self.log_folder),
                                           write_graph=False)  # quite SLOW LINE

        monitor_value = "val_loss"

        # model saving after each epoch
        checkpoint_callback = ModelCheckpoint(self.model_weights_path, monitor=monitor_value,
                                              save_weights_only=True, save_best_only=True)

        callbacks = [tensorboard_callback, checkpoint_callback]

        if early_stopping_patience >= 0:
            early_stopping = EarlyStopping(monitor=monitor_value, patience=early_stopping_patience, verbose=True)
            callbacks.append(early_stopping)

        logger.info("fitting the model...")

        if use_fit_generator:
            # to prevent memory error, only loads parts of dataset at once
            # or when using bucketing

            if bucketing:
                generator = self._training_data_bucketing(batch_size, infinite=True,
                                                          shuffle=True, bucket_range=bucket_range)
            else:
                generator = self._training_data_gen(batch_size, infinite=True,
                                                    shuffle=True)

            test_data_generator = self._test_data_gen(batch_size, infinite=True, return_steps=True)

            # first returned value from the generator is number of steps for one epoch
            steps = next(generator)
            test_steps = next(test_data_generator)

            logger.info("traning generator will make {} steps".format(steps))

            self.model.fit_generator(generator,
                                     steps_per_epoch=steps,
                                     epochs=epochs,
                                     initial_epoch=initial_epoch,
                                     callbacks=callbacks,
                                     validation_data=test_data_generator,
                                     validation_steps=test_steps
                                     )
        else:
            training_data = self._get_training_data()
            test_data = self._get_test_data()

            test_data = ([test_data["encoder_input_data"], test_data["decoder_input_data"]],
                         test_data["decoder_target_data"])

            self.model.fit(
                [
                    training_data["encoder_input_data"],
                    training_data["decoder_input_data"]
                ],
                training_data["decoder_target_data"],
                batch_size=batch_size,
                epochs=epochs,
                initial_epoch=initial_epoch,
                validation_data=test_data,
                callbacks=callbacks
            )

    def translate_test_data(self, batch_size=64, beam_size=1):
        """
        Generates translations for the test dataset, stores them in '.translated' file
        """
        logger.info("translating the test dataset...")

        steps = self.get_gen_steps(self.test_dataset, batch_size)
        logger.info("test generator will make {} steps".format(steps))

        path_original = self.test_dataset_path + "." + self.target_lang
        path = path_original + ".translated"

        step = 1
        with open(path, "w", encoding="utf-8") as out_file:
            for inputs, targets in self._test_data_gen(1, infinite=False):
                print("\rtranslating {} seq out of {}".format(step, self.test_dataset.num_samples), end="", flush=True)
                step += 1
                encoder_input_data = inputs[0]
                for i in range(len(encoder_input_data)):
                    # we need to keep the item in array ([i: i + 1])
                    decoded_sentence = self.translate_sequence_beam(encoder_input_data[i: i + 1], beam_size)

                    out_file.write(decoded_sentence + "\n")
        print("\n", end="\n")

    def get_bleu_for_test_data_translation(self):
        """
        Return BLEU score for previously generated translation (with translate_test_data)

        Returns (float): BLEU score

        """
        path_original = self.test_dataset_path + "." + self.target_lang
        path = path_original + ".translated"

        bleu = utils.get_bleu(path_original, path)

        return bleu

    def translate(self, seq=None, expected_seq=None, beam_size=1):
        """

        Translates given sequence

        Args:
            seq: sequence that will be translated from source to target language.
            expected_seq: optional, expected result of translation
            beam_size: how many candidate resulsts should be used during inference of the translated sequence for the beam search algorythm

        """

        encoded_seq = Translator.encode_text_seq_to_encoder_seq(seq, self.source_vocab)

        decoded_sentence = self.translate_sequence_beam(encoded_seq, beam_size)

        logger.info("Input sequence: {}".format(seq))
        logger.info("Expcected sentence: {}".format(expected_seq))
        print("Translated sentence: {}".format(decoded_sentence))
