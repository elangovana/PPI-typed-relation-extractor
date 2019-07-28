import logging

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

"""
Extracts vocab from data frame columns which have already been tokenised into words
"""


class TransformTextToIndex:

    def __init__(self, min_vocab_frequency=2, vocab=None):
        self.min_vocab_frequency = min_vocab_frequency
        self.vocab = vocab
        self.vocab_index = None

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def count_vectoriser(self):
        self.__count_vectoriser__ = getattr(self, "__count_vectoriser__", None) or CountVectorizer(
            vocabulary=self.vocab)
        return self.__count_vectoriser__

    @count_vectoriser.setter
    def count_vectoriser(self, value):
        self.__count_vectoriser__ = value

    def fit(self, df, Y=None):
        x = np.reshape(df.values, (df.size))
        self.count_vectoriser.fit(x)
        self.vocab_index = {key: i for i, key in enumerate(self.count_vectoriser.get_feature_names())}
        self.vocab_index["UNK"] = len(self.vocab_index)

    def transform(self, df, Y=None):
        tokeniser = self.count_vectoriser.build_tokenizer()
        df_transformed = df.applymap(lambda x: [self.vocab_index.get(w, self.vocab_index["UNK"]) for w in tokeniser(x)])

        return df_transformed

    def fit_transform(self, df, Y=None):
        self.fit(df, Y)
        return self.transform(df)
