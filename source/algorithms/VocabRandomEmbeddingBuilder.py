from sklearn.pipeline import Pipeline
from torch import nn

from algorithms.Parser import PAD
from algorithms.transform_extract_vocab import TransformExtractVocab
from algorithms.transform_tokenise import TransformTokenise

"""
Builds vocab based on training data and initiliases with random word embeddings
"""


class VocabRandomEmbeddingBuilder:

    def __init__(self, embedding_dim=200, min_vocab_frequency=1):
        self.min_vocab_frequency = min_vocab_frequency
        self.embedding_dim = embedding_dim
        self.transformer_vocab_extractor = None

    @property
    def transformer_vocab_extractor(self):
        if self.__transformer_vocab_extractor__ is None:
            # this is the default pipeline
            self.__transformer_vocab_extractor__ = Pipeline([
                ('TransformExtractWords', TransformTokenise())
                , ('TransformWordsIndices', TransformExtractVocab(min_vocab_frequency=self.min_vocab_frequency))
            ])

        return self.__transformer_vocab_extractor__

    @transformer_vocab_extractor.setter
    def transformer_vocab_extractor(self, value):
        self.__transformer_vocab_extractor__ = value

    def __call__(self, train, **kwargs):
        """
Construct vocabulary from training data
            :type train: DataFrame
            """
        # Extract train specific features
        train_specific_vocab = self.transformer_vocab_extractor.transform(train)
        # Initialise train vocab with random weights,
        rand_words_weights_dict = {}

        for word in train_specific_vocab.keys():
            # Pad character is a vector of all zeros
            if word == PAD:
                rand_words_weights_dict[word] = [0] * self.embedding_dim
            else:
                rand_words_weights_dict[word] = \
                    nn.Embedding(1, self.embedding_dim).weight.detach().numpy().tolist()[0]
        return rand_words_weights_dict
