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

    def fit(self, data_loader):
        self.logger.info("Running TransformLabelEncoder")
        y = []
        for idx, b in enumerate(data_loader):
            b_y = b[1]
            y.extend(b_y)
        self._encoder.fit(y)
        self.logger.info("Complete TransformLabelEncoder")


    def transform(self, data_loader):
        # Check if iterable
        try:
            iter(data_loader)
            iterable = not isinstance(data_loader, str)

        except TypeError:
            iterable = False

        # Looks like single value
        if not iterable:
            return self._encoder.transform([data_loader])[0]

        batches = []
        for idx, b in enumerate(data_loader):
            b_x = b[0]
            b_y = b[1]
            encoded_y = self._encoder.transform(b_y)
            batches.append([b_x, encoded_y])
        return batches

    def inverse_transform(self, Y):
        # Check if iterable
        try:
            int(Y)
            return self._encoder.inverse_transform([Y])[0]
        except TypeError:
            pass

        i = []
        for _, b in enumerate(Y):
            i.append(b)
        return self._encoder.inverse_transform(i)

    def fit_transform(self, data_loader):

        self.fit(data_loader)
        return self.transform(data_loader)
