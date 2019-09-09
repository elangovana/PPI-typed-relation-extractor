import logging

import torch
from sklearn.feature_extraction.text import CountVectorizer

"""
Extracts vocab from data frame columns which have already been tokenised into words
"""




class TransformTextToIndex:

    def __init__(self, max_feature_lens, min_vocab_frequency=2, vocab_dict=None):
        self._vocab_dict = vocab_dict or {}

        self.max_feature_lens = max_feature_lens
        self.min_vocab_frequency = min_vocab_frequency

        # Load pretrained vocab

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def construct_vocab_dict(self, data_loader):
        return self._get_vocab_dict(data_loader)

    @staticmethod
    def pad_token():
        return "!@#"

    @property
    def vocab_dict(self):
        return self._vocab_dict

    @vocab_dict.setter
    def vocab_dict(self, vocab_index):
        self._vocab_dict = vocab_index


    def fit(self, data_loader):
        if self._vocab_dict is None or len(self._vocab_dict) == 0:
            self._vocab_dict = self._get_vocab_dict(data_loader)

    @staticmethod
    def _get_vocab_dict(data_loader):
        count_vectoriser = CountVectorizer()
        for idx, b in enumerate(data_loader):
            b_x = b[0]

            text = [" ".join(t) for t in b_x]
            count_vectoriser.fit(text)

        vocab_index = count_vectoriser.vocabulary_

        vocab_index[TransformTextToIndex.pad_token()] = len(vocab_index)
        vocab_index["UNK"] = len(vocab_index)
        return vocab_index

    def transform(self, x):
        self.logger.info("Transforming TransformTextToIndex")
        tokeniser = CountVectorizer().build_tokenizer()
        pad_index = self._vocab_dict[self.pad_token()]

        batches = []
        for idx, b in enumerate(x):
            b_x = b[0]
            b_y = b[1]
            col = []
            for c_index, c in enumerate(b_x):
                row = []
                max = self.max_feature_lens[c_index]
                for _, r in enumerate(c):
                    tokens = [self._vocab_dict.get(w, self._vocab_dict["UNK"]) for w in tokeniser(r)][0:max]
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
