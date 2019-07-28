import logging

import numpy
from torchtext import data

from algorithms.Parser import Parser, PAD

"""
Constraucts examples
"""


class TransformFinalCreateExamples:

    def __init__(self):
        self.parser = None

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def parser(self):
        self.__parser__ = self.__parser__ or Parser()
        return self.__parser__

    @parser.setter
    def parser(self, value):
        self.__parser__ = value

    def fit(self, df, y=None):
        self.feature_lens = df.apply(
            lambda c: numpy.math.ceil(numpy.percentile(c.apply(len), 90))).values
        self.logger.info("Column length counts : {}".format(self.feature_lens))

    def transform(self, df, y=None):
        col_names = df.columns.values
        data_formatted = self._getexamples(col_names, self.feature_lens, df, y)
        return data_formatted

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df, y)

    def _getexamples(self, col_names, feature_lens, df, labels=None):
        fields = [(c, data.Field(use_vocab=False, pad_token=self.parser.get_min_dictionary()[PAD], fix_length=l)) for
                  c, l in
                  zip(col_names, feature_lens)]

        if labels is not None:
            LABEL = data.LabelField(use_vocab=False, sequential=False, is_target=True)
            fields.append(("label", LABEL))
            return data.Dataset([data.Example.fromlist([*f, l], fields) for l, f in
                                 zip(labels, df.values.tolist())], fields)

        return data.Dataset([data.Example.fromlist([*f], fields) for f in
                             df.values.tolist()], fields)
