import logging

import torch
from sklearn.feature_extraction.text import CountVectorizer

"""
Extracts vocab from data frame columns which have already been tokenised into words
"""


class TransformTextToIndex:

    def __init__(self, max_feature_lens, min_vocab_doc_frequency=4, case_insensitive=True, vocab_dict=None,
                 special_words=None, use_dataset_vocab=True):
        self.use_dataset_vocab = use_dataset_vocab
        self.case_insensitive = case_insensitive
        self.special_words = special_words or []
        self._vocab_dict = vocab_dict or {}

        self.max_feature_lens = max_feature_lens
        self.min_vocab_doc_frequency = min_vocab_doc_frequency

        # Load pretrained vocab

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def construct_vocab_dict(self, data_loader):
        if self.use_dataset_vocab:
            vocab = self._get_vocab_dict(data_loader, self.special_words, case_insensitive=self.case_insensitive,
                                         min_vocab_doc_frequency=self.min_vocab_doc_frequency)
        else:
            vocab = self.get_specialwords_dict()

        self.logger.info("Constructed vocab of size {}".format(len(vocab)))

        return vocab

    def get_specialwords_dict(self):
        tokens = [self.pad_token(), self.eos_token(), "PROTEIN1", "PROTEIN2", "PROTEIN_1",
                  "PROTEIN_2", self.UNK_token()]
        key_func = lambda x: x
        if self.case_insensitive:
            key_func = lambda x: x.lower()
        tokens_dict = {key_func(k): i for i, k in enumerate(tokens)}
        return tokens_dict

    @staticmethod
    def pad_token():
        return "!@#"

    @staticmethod
    def eos_token():
        return "<EOS>"

    @staticmethod
    def UNK_token():
        return "<UNK>"

    @property
    def vocab_dict(self):
        return self._vocab_dict

    @vocab_dict.setter
    def vocab_dict(self, vocab_index):
        self._vocab_dict = vocab_index

    def fit(self, data_loader):
        if self._vocab_dict is None or len(self._vocab_dict) == 0:
            self._vocab_dict = self._get_vocab_dict(data_loader, self.special_words, self.case_insensitive,
                                                    self.min_vocab_doc_frequency)

    @staticmethod
    def _get_vocab_dict(data_loader, special_words, case_insensitive, min_vocab_doc_frequency):
        count_vectoriser = CountVectorizer(lowercase=case_insensitive, min_df=min_vocab_doc_frequency)
        f = lambda x: x
        if case_insensitive:
            f = lambda x: x.lower()

        # fit pad first so that it has index zero

        text = []
        for idx, b in enumerate(data_loader):
            b_x = b[0]

            column_text = [" ".join(t) for t in b_x]
            text.extend(column_text)

        count_vectoriser.fit(text)


        vocab_index = count_vectoriser.vocabulary_

        # Set up so that the vocab of pad token
        vocab_index = {k: v + 1 for k, v in vocab_index.items()}
        vocab_index[f(TransformTextToIndex.pad_token())] = 0

        vocab_index[f(TransformTextToIndex.UNK_token())] = vocab_index.get(f(TransformTextToIndex.UNK_token()),
                                                                           len(vocab_index))
        vocab_index[(TransformTextToIndex.eos_token())] = vocab_index.get(f(TransformTextToIndex.eos_token()),
                                                                          len(vocab_index))
        for w in set(special_words):
            vocab_index[f(w)] = vocab_index.get(f(w), len(vocab_index))

        return vocab_index

    def transform(self, x):
        self.logger.info("Transforming TransformTextToIndex")
        f = lambda x: x
        if self.case_insensitive:
            f = lambda x: x.lower()

        tokeniser = CountVectorizer().build_tokenizer()
        pad_index = self._vocab_dict[f(self.pad_token())]

        batches = []
        for idx, b in enumerate(x):
            b_x = b[0]
            b_y = b[1]
            col = []
            for c_index, c in enumerate(b_x):
                row = []
                max = self.max_feature_lens[c_index]
                for _, r in enumerate(c):
                    tokens = [self._vocab_dict.get(f(w), self._vocab_dict[f(TransformTextToIndex.UNK_token())]) for w in
                              tokeniser(r)][0:max]
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
