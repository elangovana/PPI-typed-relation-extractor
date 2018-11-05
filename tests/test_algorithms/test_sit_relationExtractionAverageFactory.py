import os
import tempfile
from io import StringIO
from logging.config import fileConfig
from unittest import TestCase

import pandas as pd

from algorithms.RelationExtractionAverageFactory import RelationExtractionAverageFactory


class TestSitRelationExtractionAverageFactory(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_call(self):
        # Arrange
        out_dir = tempfile.mkdtemp()
        embedding = StringIO(
            "\n".join(["hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9"]))
        sut = RelationExtractionAverageFactory(class_size=2, embedding_handle=embedding, embedding_dim=3, ngram=1,
                                               output_dir=out_dir)

        train_df = [["This is good", "entity1", "entity2"],
                    ["this is a cat not a hat", "mat protein", "cat protein"]]

        val_data = [["This is hat", "entity1", "entity2"],
                    ["this is a cat not a mat", "mat protein", "cat protein"]]

        labels = ["1", "0"]
        cols = ['abstract', 'entity1', 'entity2']
        train_df = pd.DataFrame(train_df, columns=cols)
        val_df = pd.DataFrame(val_data, columns=cols)

        # Act
        actual = sut(train_df, labels, val_df, labels)
