import logging

import numpy

from algorithms.Parser import UNKNOWN_WORD, EOS

"""
Extracts words
"""


class TransformTokensToIndices:

    def __init__(self, vocab, add_eos=False):
        self.add_eos = add_eos
        self.vocab = vocab

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.applymap(lambda x: self._make_array(x, self.vocab))
        return df

    def _make_array(self, tokens, vocab):
        unk_id = vocab[UNKNOWN_WORD]
        eos_id = vocab[EOS]
        ids = [vocab.get(token, unk_id) for token in tokens]
        if self.add_eos:
            ids.append(eos_id)
        return numpy.array(ids, numpy.int32)
