import logging

from torchtext import data

from algorithms.Parser import Parser, PAD

"""
Constraucts examples
"""


class TransformFinalCreateExamples:

    def __init__(self, feature_lens):
        self.feature_lens = feature_lens
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

    def fit(self, X, y=None):
        return self

    def transform(self, df, y=None):
        col_names = df.columns.values

        data_formatted = self._getexamples(col_names, self.feature_lens, df, y)

        return data_formatted

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
