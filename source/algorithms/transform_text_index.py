import logging

import torch
from sklearn.feature_extraction.text import CountVectorizer

"""
Extracts vocab from data frame columns which have already been tokenised into words
"""


class TransformTextToIndex:

    def __init__(self, max_feature_lens, min_vocab_frequency=2, vocab=None):
        self.max_feature_lens = max_feature_lens
        self.min_vocab_frequency = min_vocab_frequency
        self.vocab = vocab
        self.vocab_index = None
        self.pad = "!@#"

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

    def fit(self, data_loader):
        for idx, b in enumerate(data_loader):
            b_x = b[0]

            text = [" ".join(t) for t in b_x]
            self.count_vectoriser.fit(text)
        self.vocab = self.count_vectoriser.get_feature_names()
        self.vocab_index = {key: i for i, key in enumerate(self.vocab)}
        self.vocab_index["UNK"] = len(self.vocab_index)
        self.vocab_index[self.pad] = len(self.vocab_index)

    def transform(self, x):
        self.logger.info("Transforming TransformTextToIndex")
        tokeniser = self.count_vectoriser.build_tokenizer()
        pad_index = self.vocab_index[self.pad]

        batches = []
        for idx, b in enumerate(x):
            b_x = b[0]
            b_y = b[1]
            col = []
            for c_index, c in enumerate(b_x):
                row = []
                max = self.max_feature_lens[c_index]
                for _, r in enumerate(c):
                    tokens = [self.vocab_index.get(w, self.vocab_index["UNK"]) for w in tokeniser(r)][0:max]
                    tokens = tokens + [pad_index] * (max - len(tokens))
                    row.append(tokens)
                row = torch.Tensor(row).long()
                col.append(row)

            batches.append([col, b_y])
        self.logger.info("Completed TransformTextToIndex")
        return batches

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)
