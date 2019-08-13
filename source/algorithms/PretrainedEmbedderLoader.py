import logging

import numpy as np
from torch import nn


class PretrainedEmbedderLoader:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def get_vocab_dict(self, handle):
        vocab_dict = {}
        for i, line in enumerate(handle):
            # skip first line as it contains just the dim
            if i == 0: continue
            values = line.split()
            word = values[0]
            vocab_dict[word] = vocab_dict.get(word, len(vocab_dict))
        return vocab_dict

    def __call__(self, handle, initial_words_index_dict=None):
        """
Expects the stream of strings to contain word embedding. Each record must be in space separated format with the first column containing the word itself.
Each record is separated by new lines
e.g for an embedding of size 10
zsombor -0.75898 -0.47426 0.4737 0.7725 -0.78064 0.23233 0.046114 0.84014 0.243710 .022978
sandberger 0.072617 -0.51393 0.4728 -0.52202 -0.35534 0.34629 0.23211 0.23096 0.26694 .41028
        :param words_index_dict: The index of words so the correct embedding is assigned
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

        embeddings_array = [[]] * len(initial_words_index_dict)

        result_words_index_dict = {}
        # Load embeddings from file
        for i, line in enumerate(handle):
            # skip first line as it contains just the dim
            if i == 0: continue
            values = line.split()
            word = values[0]

            # Not ignored word
            embeddings = [float(v) for v in values[1:]]
            if word not in initial_words_index_dict:
                result_words_index_dict[word] = len(embeddings_array)
                embeddings_array.append([])
            else:
                result_words_index_dict[word] = initial_words_index_dict[word]

            embeddings_array[result_words_index_dict[word]] = embeddings

        embedding_dim = len(embeddings_array[0])
        words_not_in_embedding = []
        for k in initial_words_index_dict:
            # Word not found in embedding, init with random
            if embeddings_array[initial_words_index_dict[k]] == []:
                result_words_index_dict[k] = initial_words_index_dict[k]
                words_not_in_embedding.append(k)
                embeddings_array[initial_words_index_dict[k]] = \
                    nn.Embedding(1, embedding_dim).weight.detach().numpy().tolist()[
                        0]

        self.logger.info("The number of words intialised without embbeder is {}".format(len(words_not_in_embedding)))
        self.logger.debug("The words intialised without embbeder is \n {}".format(words_not_in_embedding))

        # Convert to ndarray or cupy
        embeddings_array = np.array(embeddings_array)
        print(embeddings_array)
        print(result_words_index_dict)
        return result_words_index_dict, embeddings_array
