from unittest import TestCase
from unittest.mock import Mock

import pandas as pd

from algorithms.VocabRandomEmbeddingBuilder import VocabRandomEmbeddingBuilder


class TestVocabRandomEmbeddingBuilder(TestCase):

    def test___call__(self):
        # Arrange
        # set up input data
        train_data = [["This is a protein"], ["This is a gene"]]
        data_df = pd.DataFrame(train_data)
        tokens = {"protein": 0, "this": 1, "gene": 2}

        # Set up sut
        embed_dim = 5
        sut = VocabRandomEmbeddingBuilder(embedding_dim=embed_dim)
        mock_vocab_extractor = Mock()
        mock_vocab_extractor.transform.return_value = tokens
        sut.transformer_vocab_extractor = mock_vocab_extractor

        # Act
        actual_embedding = sut(data_df)

        # Assert
        self.assertEqual(len(actual_embedding), len(tokens),
                         "The length of the embedding should match the number of tokens")
