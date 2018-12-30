from io import StringIO
from unittest import TestCase
from unittest.mock import Mock

import pandas as pd

from algorithms.VocabRandomPretrainedEmbedCombiner import VocabRandomPretrainedEmbedCombiner


class TestVocabRandomPretrainedEmbedCombiner(TestCase):

    def test___call__(self):
        # Arrange
        # set up input data
        train_data = [["This is a protein"], ["This is a gene"]]
        data_df = pd.DataFrame(train_data)

        # set up mocks
        mock_pretrained_loader = Mock()
        mock_pretrained_loader.return_value = ({"protein": 0, "phosphate": 1}, [[.2, .3], [.3, .3]])
        mock_vocab_random_builder = Mock()
        mock_vocab_random_builder.return_value = ({"gene": 0, "protein": 1}, [[.1, .4], [.9, .8]])

        # set up expected values
        expected_vocab_keys = set(mock_pretrained_loader.return_value[0]).union(
            mock_vocab_random_builder.return_value[0])
        expected_embeddings_dict = {"protein": [.2, .3], "phosphate": [.3, .3], "gene": [.1, .4]}

        # set up sut
        handle = StringIO()
        sut = VocabRandomPretrainedEmbedCombiner(embedding_handle=handle)
        sut.pretrained_embedder_loader = mock_pretrained_loader
        sut.vocab_random_builder = mock_vocab_random_builder

        # Act
        actual_vocab, actual_embedding = sut(data_df)

        # Assert
        self.assertEqual(set(actual_vocab.keys()), expected_vocab_keys,
                         "The vocab keys should be a combination of the tokens in pretrained embeddings and the vocab builder")
        self.assertEqual(len(actual_vocab), len(actual_embedding),
                         "The length of vocab should match length of embeddings")
        for w, i in actual_vocab.items():
            self.assertEqual(actual_embedding[i], expected_embeddings_dict[w],
                             "The index of the token {} should line up to the index {} in the embedding".format(w, i))
