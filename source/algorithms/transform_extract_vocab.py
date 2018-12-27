import itertools
import logging
from collections import Counter

from algorithms.Parser import Parser

"""
Extracts vocab from data frame columns which have already been tokenised into words
"""


class TransformExtractVocab:

    def __init__(self, min_vocab_frequency=2):
        self.min_vocab_frequency = min_vocab_frequency

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        vocab = self.construct_vocab(df)
        return vocab

    def construct_vocab(self, train_data):
        vocab_token_counter = Counter()
        for i in train_data.apply(lambda c: self.get_column_values_count(c), axis=0).values:
            vocab_token_counter += i

        result = Parser.get_min_dictionary()
        for ik, iv in filter(lambda t: t[1] >= self.min_vocab_frequency, vocab_token_counter.items()):
            result[ik] = len(result)

        return result

    def get_column_values_count(self, c):
        values = list(itertools.chain.from_iterable(c.values))
        return Counter(values)
