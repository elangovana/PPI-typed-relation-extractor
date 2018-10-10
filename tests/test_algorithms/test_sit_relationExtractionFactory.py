from io import StringIO
from unittest import TestCase

from algorithms.RelationExtractionFactory import RelationExtractionFactory


class TestSitRelationExtractionFactory(TestCase):
    def test_call(self):
        # Arrange
        embedding = StringIO("\n".join(["hat 0.2 .34", "mat 0.5 .34"]))
        sut = RelationExtractionFactory(class_size=2, embedding_handle=embedding, embedding_dim=2)
        data = [("This is good", "0"), ( "This is a hat, but not a cat", "1")]

        # Act
        sut(data)
