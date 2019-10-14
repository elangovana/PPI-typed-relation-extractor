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
        expected_embedding_len = 1

        embed_dim = 3
        sut = PretrainedEmbedderLoaderMinimum(pad, dim=embed_dim)

        # Act
        word_index_dict, embeddings_array = sut(embedding_handle, initial_words_index_dict={pad: 0})

        # Assert
        self.assertEqual(embeddings_array.shape[0], expected_embedding_len)

    def test___call__worddict_wordexists(self):
        """
        Case where dict exists in embedding
        """
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
        embed_dim = 3

        word_dict = {pad: 0, "entity1": 1}

        expected_embedding = [[0.0, 0.0, 0.0], [0.5, .55, 0.8, ]]
        expected_vocab = {pad: 0, "entity1": 1}

        sut = PretrainedEmbedderLoaderMinimum(pad, dim=embed_dim)

        # Act
        word_index_dict, embeddings_array = sut(embedding_handle, initial_words_index_dict=word_dict)

        # Assert
        self.assertEqual(expected_vocab, word_index_dict)
        self.assertSequenceEqual(expected_embedding, embeddings_array.tolist())

    def test___call__worddict_nonexistent_word(self):
        """
        Case where dict words doesnt exist in embedding
        """
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
        embed_dim = 3

        word_dict = {pad: 0, "entityX": 1}

        expected_vocab = {pad: 0, "entityX": 1}

        sut = PretrainedEmbedderLoaderMinimum(pad, dim=embed_dim)

        # Act
        word_index_dict, embeddings_array = sut(embedding_handle, initial_words_index_dict=word_dict)

        # Assert
        self.assertEqual(expected_vocab, word_index_dict)
        self.assertEqual(embeddings_array.shape[0], len(word_index_dict))
