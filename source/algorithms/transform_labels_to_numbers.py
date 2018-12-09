import logging

from sklearn import preprocessing

"""
Extracts vocab from data frame columns which have already been tokenised into words
"""


class TransformLabelsToNumbers:

    def __init__(self, classes):
        self.classes = classes

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, y):
        transformed_labels = self.encode_labels(y, self.classes)
        return transformed_labels

    def encode_labels(self, y, classes):
        le = preprocessing.LabelEncoder()
        le.classes_ = classes
        return le.transform(y)
