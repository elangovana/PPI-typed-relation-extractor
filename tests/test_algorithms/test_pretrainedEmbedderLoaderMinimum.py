from io import StringIO
from unittest import TestCase

from algorithms.PretrainedEmbedderLoaderMinimum import PretrainedEmbedderLoaderMinimum


class TestPretrainedEmbedderLoaderMinimum(TestCase):
    def test___call__simplecase(self):
        embeddings_list = ["4 3",
                           "hat 0.2 .34 0.8",
                           "mat 0.5 .34 0.8",
                           "entity1 0.5 .55 0.8",
                           "entity2 0.3 .55 0.9"]
        embedding_handle = StringIO(
            "\n".join(embeddings_list))
        pad = "[#$%]"
        expected_embedding_len = 0

        sut = PretrainedEmbedderLoaderMinimum(pad)

        # Act
        word_index_dict, embeddings_array = sut(embedding_handle, initial_words_index_dict=None)

        # Assert
        self.assertEqual(embeddings_array.shape[0], expected_embedding_len)

    def test___call__worddict(self):
        # Arrange
        # Construct embedding
        embeddings_list = ["4 3",
                           "hat 0.2 .34 0.8",
                           "mat 0.5 .34 0.8",
                           "entity1 0.5 .55 0.8",
                           "entity2 0.3 .55 0.9"]
        embedding_handle = StringIO(
            "\n".join(embeddings_list))
        pad = "[#$%]"
        existing_word_dict = {"entity1": 0}

        expected_embedding = [[0.5, .55, 0.8, ]]
        expected_vocab = {"entity1": 0}

        sut = PretrainedEmbedderLoaderMinimum(pad)

        # Act
        word_index_dict, embeddings_array = sut(embedding_handle, initial_words_index_dict=existing_word_dict)

        # Assert
        self.assertEqual(expected_vocab, word_index_dict)
        self.assertSequenceEqual(expected_embedding, embeddings_array.tolist())
