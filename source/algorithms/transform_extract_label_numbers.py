import logging

from sklearn import preprocessing

"""
Extracts unique labels 
"""


class TransformExtractLabelNumbers:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, y):
        labels = self._get_label_map(y)
        return labels

    def _get_label_map(self, y):
        le = preprocessing.LabelEncoder()
        le.fit(y)
        return le.classes_
