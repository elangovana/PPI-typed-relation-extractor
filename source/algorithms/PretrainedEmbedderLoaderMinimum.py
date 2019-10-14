import logging

import numpy as np


class PretrainedEmbedderLoaderMinimum:
    """
    Only uses words in  vocab
    """

    def __init__(self, pad_token, dim):
        self.dim = dim
        self.pad_token = pad_token

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, handle, initial_words_index_dict=None):
        """
Expects the stream of strings to contain word embedding. Each record must be in space separated format with the first column containing the word itself.
Each record is separated by new lines
e.g for an embedding of size 10
zsombor -0.75898 -0.47426 0.4737 0.7725 -0.78064 0.23233 0.046114 0.84014 0.243710 .022978
sandberger 0.072617 -0.51393 0.4728 -0.52202 -0.35534 0.34629 0.23211 0.23096 0.26694 .41028
        :param words_index_dict: The index of words to filter out
        :return: a tuple (word_index_dict, embeddings_array)
        :param handle: handle containing the embedding
        """
        initial_words_index_dict = initial_words_index_dict or {}

        if len(initial_words_index_dict) > 0:
            assert len(initial_words_index_dict) - 1 == max(
                initial_words_index_dict.values()), "The word index dict must be values between 0 to len(dict) {}, but found max value to be {}".format(
                len(initial_words_index_dict) - 1,
                max(initial_words_index_dict.values()))
            assert min(initial_words_index_dict.values()) == 0, "The word index dict must be zero indexed values"

        embeddings_array = 0.002 * np.random.random_sample((len(initial_words_index_dict), self.dim)) - 0.001

        embeddings_array[initial_words_index_dict[self.pad_token]] = 0.

        result_words_index_dict = initial_words_index_dict.copy()
        total_embed_words = 0
        total_random_init_word = 0
        # Load embeddings from file
        for i, line in enumerate(handle):
            # skip first line as it contains just the dim
            if i == 0: continue
            values = line.split(" ")
            word = values[0]

            # Not ignored word
            embeddings = [float(v) for v in values[1:]]

            if word in result_words_index_dict:
                embeddings_array[result_words_index_dict[word]] = embeddings
            else:
                total_random_init_word += 1
            total_embed_words = i

        self.logger.info("Total words in original embedding handle is {}".format(total_embed_words))
        self.logger.info("Total words in final embedding is {}".format(len(result_words_index_dict)))
        self.logger.info("Total words randomly initialized is {}".format(total_random_init_word))

        # Convert to ndarray or cupy
        embeddings_array = np.array(embeddings_array)

        return result_words_index_dict, embeddings_array
