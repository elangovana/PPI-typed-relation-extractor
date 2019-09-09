from io import StringIO
from unittest import TestCase

from algorithms.PretrainedEmbedderLoader import PretrainedEmbedderLoader


class TestPretrainedEmbedderLoader(TestCase):
    def test___call__simplecase(self):
        embed_dim = 3
        embeddings_list = ["4 3",
                           "hat 0.2 .34 0.8",
                           "mat 0.5 .34 0.8",
                           "entity1 0.5 .55 0.8",
                           "entity2 0.3 .55 0.9"]
        embedding_handle = StringIO(
            "\n".join(embeddings_list))
        pad = "[#$%]"

        sut = PretrainedEmbedderLoader(pad)

        # Act
        word_index_dict, embeddings_array = sut(embedding_handle, initial_words_index_dict=None)

        # Assert
        self.assertEqual(embeddings_array.shape[0], len(word_index_dict))
        self.assertEqual(len(embeddings_list) - 1, len(word_index_dict))
        self.assertEqual((len(embeddings_list) - 1, embed_dim), embeddings_array.shape)
        self.assertSequenceEqual(embeddings_array[word_index_dict["hat"]].tolist(), [0.2, .34, 0.8])

    def test___call__worddict(self):
        # Arrange
        # Construct embedding
        embed_dim = 3
        embeddings_list = ["4 3",
                           "hat 0.2 .34 0.8",
                           "mat 0.5 .34 0.8",
                           "entity1 0.5 .55 0.8",
                           "entity2 0.3 .55 0.9"]
        embedding_handle = StringIO(
            "\n".join(embeddings_list))
        pad = "[#$%]"
        existing_word_dict = {"entity1": 0}

        embed_words_size = len(embeddings_list) - 1

        sut = PretrainedEmbedderLoader(pad)

        # Act
        word_index_dict, embeddings_array = sut(embedding_handle, initial_words_index_dict=existing_word_dict)

        # Assert
        self.assertEqual(embeddings_array.shape[0], len(word_index_dict))
        self.assertEqual(embed_words_size, len(word_index_dict))
        self.assertEqual((embed_words_size, embed_dim), embeddings_array.shape)
        self.assertSequenceEqual(embeddings_array[existing_word_dict["entity1"]].tolist(), [0.5, .55, 0.8])

    def test___call__worddict_with_new_words(self):
        """
        Test with new words not in embedding
        :return:
        """
        embed_dim = 3
        embeddings_list = ["4 3",
                           "hat 0.2 .34 0.8",
                           "mat 0.5 .34 0.8",
                           "entity1 0.5 .55 0.8",
                           "entity2 0.3 .55 0.9"]
        embedding_handle = StringIO(
            "\n".join(embeddings_list))
        pad = "[#$%]"
        # Additional dictionary passed in with words not found in the embeddings handle
        existing_word_dict = {"entity1": 0, "newword": 1}

        embed_words_size = len(embeddings_list) - 1
        # One new word in the existing word dictionary
        total_words = embed_words_size + 1

        sut = PretrainedEmbedderLoader(pad)

        # Act
        word_index_dict, embeddings_array = sut(embedding_handle, initial_words_index_dict=existing_word_dict)

        # Assert
        self.assertEqual(total_words, len(word_index_dict))
        self.assertEqual((total_words, embed_dim), embeddings_array.shape)
        self.assertEqual(embed_dim, len(embeddings_array[existing_word_dict["newword"]].tolist()))

    def test___call__padembedding(self):
        """
        Test with new words not in embedding
        :return:
        """
        embed_dim = 3
        embeddings_list = ["4 3",
                           "hat 0.2 .34 0.8",
                           "mat 0.5 .34 0.8",
                           "entity1 0.5 .55 0.8",
                           "entity2 0.3 .55 0.9"]
        embedding_handle = StringIO(
            "\n".join(embeddings_list))
        pad = "[#$%]"
        # Additional dictionary pad
        existing_word_dict = {pad: 0}

        embed_words_size = len(embeddings_list) - 1
        # One new word in the existing word dictionary
        total_words = embed_words_size + 1

        sut = PretrainedEmbedderLoader(pad)

        # Act
        word_index_dict, embeddings_array = sut(embedding_handle, initial_words_index_dict=existing_word_dict)

        # Assert
        self.assertEqual([0] * embed_dim, embeddings_array[existing_word_dict[pad]].tolist())
