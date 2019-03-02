"""
Converts a torchtext iterable of examples to dataframe.
"""
import pandas as pd


class ExampleDataFrameConverter:

    def __init__(self):
        pass

    def to_df(self, dataset):
        """
Converts torchtext.data.Dataset object to a pandas DataFrame
        :type dataset: torchtext.data.Dataset
        """
        items = []
        for r in dataset:
            items.append(self._to_dict(r))

        result = pd.DataFrame(items)
        return result

    def _to_dict(self, r):
        result = {}
        for attr, value in r.__dict__.items():
            result[attr] = value

        return result
