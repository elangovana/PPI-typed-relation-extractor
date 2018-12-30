import logging

from algorithms.PretrainedEmbedderLoader import PretrainedEmbedderLoader
from algorithms.VocabRandomEmbeddingBuilder import VocabRandomEmbeddingBuilder

"""
Combines the training data vocab with pretrained embeddings
"""


class VocabRandomPretrainedEmbedCombiner:

    def __init__(self, embedding_handle):
        self.embedding_handle = embedding_handle
        self.pretrained_embedder_loader = None
        self.vocab_random_builder = None

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def vocab_random_builder(self):
        self.__vocab_random_builder__ = self.__vocab_random_builder__ or VocabRandomEmbeddingBuilder(
            min_vocab_frequency=2)
        return self.__vocab_random_builder__

    @vocab_random_builder.setter
    def vocab_random_builder(self, value):
        self.__vocab_random_builder__ = value

    @property
    def pretrained_embedder_loader(self):
        self.__pretrained_embedder_loader__ = self.__pretrained_embedder_loader__ or PretrainedEmbedderLoader()
        return self.__pretrained_embedder_loader__

    @pretrained_embedder_loader.setter
    def pretrained_embedder_loader(self, value):
        self.__pretrained_embedder_loader__ = value

    def __call__(self, train):

        self.logger.info("Loading pretrained embedding..")
        # The full vocab is a combination of train and embeddings..
        embed_vocab, embedding_array = self.pretrained_embedder_loader(self.embedding_handle, {})

        # Extract vocab words initialised with random embeddings
        self.vocab_random_builder.embedding_dim = len(embedding_array[0])
        token_vocab, rand_weights = self.vocab_random_builder(train)
        self.logger.info("The train vocab len is {}".format(len(token_vocab)))

        debug_words_not_in_embedding = set()
        for w, i in token_vocab.items():
            if embed_vocab.get(w, None) is None:
                embed_vocab[w] = len(embed_vocab)
                embedding_array.append(rand_weights[i])
                debug_words_not_in_embedding.add(w)

        self.logger.info(
            "The number of words intialised without embedder is {}".format(len(debug_words_not_in_embedding)))
        self.logger.debug("The words intialised without embedder is \n {}".format(debug_words_not_in_embedding))

        self.logger.info("The full vocab len is {}".format(len(embed_vocab)))

        return embed_vocab, embedding_array
