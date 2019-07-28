from io import StringIO
from unittest import TestCase

import pandas as pd

from algorithms.DataPipeline import DataPipeline


class TestDataPipeline(TestCase):

    def test_fit(self):
        # Arrange
        embedding = StringIO(
            "\n".join(["hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9"]))

        cols = ['abstract', 'entity1', 'entity2']
        train = [["This is good", "entity1", "entity2"],
                 ["this is a cat not a hat", "mat protein", "cat protein"]]
        label = [True, False]

        train_df = pd.DataFrame(train, columns=cols)

        sut = DataPipeline(embeddings_handle=embedding)

        # Act
        actual_x, y = sut.fit_transform(train_df, label)

        # Assert
        self.assertEqual(len(actual_x), train_df.shape[0])
