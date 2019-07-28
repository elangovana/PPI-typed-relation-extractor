import logging

from sklearn import preprocessing

"""
Extracts vocab from data frame columns which have already been tokenised into words
"""


class TransformLabelEncoder:

    def __init__(self):
        self._encoder = preprocessing.LabelEncoder()

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, x, y=None):
        self._encoder.fit(x)

    def transform(self, x, y=None):
        return self._encoder.transform(x)

    def inverse_transform(self, Y):
        return self._encoder.inverse_transform(Y)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
