from io import StringIO
from unittest import TestCase

from algorithms.RelationExtractionFactory import RelationExtractionFactory
import pandas as pd


class TestSitRelationExtractionFactory(TestCase):
    def test_call(self):
        # Arrange
        embedding = StringIO("\n".join(["hat 0.2 .34", "mat 0.5 .34", "entity1 0.5 .55", "entity2 0.3 .55"]))
        sut = RelationExtractionFactory(class_size=2, embedding_handle=embedding, embedding_dim=2, ngram=1)

        data = [["This is good", "entity1", "entity2"],
                ["this is a cat not a hat", "mat protein", "cat protein"]]

        labels = ["1", "0"]

        data = pd.DataFrame(data)

        # Act
        actual = sut(data, labels=labels)
