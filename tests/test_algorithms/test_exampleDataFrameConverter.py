from unittest import TestCase

import pandas as pd
import torchtext
from torchtext.data import Dataset

from algorithms.ExampleDataFrameConverter import ExampleDataFrameConverter


class TestExampleDataFrameConverter(TestCase):

    def test_to_df(self):
        # Arrange
        sut = ExampleDataFrameConverter()
        fields = [("abstract", torchtext.data.Field()), ("entity1", torchtext.data.Field())]
        # data
        data_set = [torchtext.data.Example.fromlist(["This is great", "e1"], fields=fields)
            , torchtext.data.Example.fromlist(["This is awesome", "e1"], fields=fields)]
        field_names = [t[0] for t in fields]
        # Act
        actual = sut.to_df(Dataset(data_set, fields))

        # Assert
        # Return dataframe
        self.assertIsInstance(actual, pd.DataFrame)
        # Match shape
        self.assertEqual(actual.shape, (len(data_set), len(fields)))
        # Match fields
        self.assertSequenceEqual(actual.columns.values.tolist(), field_names)
