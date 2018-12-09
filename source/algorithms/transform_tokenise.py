import itertools
import logging
from collections import Counter

from nltk import wordpunct_tokenize

"""
Extracts words
"""


class TransformTokenise:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.applymap(lambda x: self.split_text(x))
        token_counts = df.apply(lambda c: self._get_column_values_count(c), axis=0).values

        self.logger.info("Token counts : {}".format(token_counts))
        return df

    def split_text(self, text, char_based=False):
        if char_based:
            return list(text)
        else:
            return wordpunct_tokenize(text)

    def _get_column_values_count(self, c):
        values = list(itertools.chain.from_iterable(c.values))
        return Counter(values)
