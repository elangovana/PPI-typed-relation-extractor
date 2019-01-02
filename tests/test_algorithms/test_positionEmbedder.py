from unittest import TestCase

import numpy as np
from ddt import ddt

from algorithms.PositionEmbedder import PositionEmbedder


@ddt
class TestPositionEmbedder(TestCase):

    def test__call__case_one_entity_occurance(self):
        """
Tests the returned position when there is just one entity
        """
        # Arrange
        max_pos = 30
        pos_dim = 2
        position_embeddings = np.random.uniform(0, 1, (max_pos, pos_dim))
        # Input data
        tokens_array = range(0, 5)
        entity = 0
        # Expected output
        expected = np.asarray([position_embeddings[i] for i in tokens_array])

        # Act
        actual = self._invoke_sut__call__(entity, position_embeddings, tokens_array)

        # Assert
        self.assertEqual(actual.shape, expected.shape)
        self.assertSequenceEqual(actual.ravel().tolist(), expected.ravel().tolist())

    def test__call__case_multiple_entity_occurance(self):
        """
Tests the returned distance is the distance to the nearest entity
        """
        # Arrange
        max_pos = 30
        pos_dim = 2
        position_embeddings = np.random.uniform(0, 1, (max_pos, pos_dim))
        # Input data
        tokens_array = [0, 3, 4, 0]
        entity = 0
        # Expected output
        expected = np.asarray(
            [position_embeddings[0], position_embeddings[1], position_embeddings[1], position_embeddings[0]])

        # Act
        actual = self._invoke_sut__call__(entity, position_embeddings, tokens_array)

        # Assert
        self.assertEqual(actual.shape, expected.shape)
        self.assertSequenceEqual(actual.ravel().tolist(), expected.ravel().tolist())

    def test__call__case_distance_higher_than_dim(self):
        """
Tests the returned position is the max distance if the distance is larger than  the embedding dim
        """
        # Arrange
        max_pos = 2
        pos_dim = 2
        position_embeddings = np.random.uniform(0, 1, (max_pos, pos_dim))
        # Input data
        tokens_array = range(0, max_pos + 1)
        entity = 0
        # Expected output
        expected = np.asarray(
            [position_embeddings[0], position_embeddings[1], position_embeddings[1]])

        # Act
        actual = self._invoke_sut__call__(entity, position_embeddings, tokens_array)

        # Assert
        self.assertEqual(actual.shape, expected.shape)
        self.assertSequenceEqual(actual.ravel().tolist(), expected.ravel().tolist())

    def test__call__case_no_matching_entity(self):
        """
Tests case where the entity is not present in the text
        """
        # Arrange
        max_pos = 2
        pos_dim = 2
        position_embeddings = np.random.uniform(0, 1, (max_pos, pos_dim))
        # Input data
        tokens_array = range(0, max_pos)
        entity = max_pos + 10
        # Expected output
        expected = np.asarray(
            [position_embeddings[max_pos - 1]] * pos_dim)

        # Act
        actual = self._invoke_sut__call__(entity, position_embeddings, tokens_array)

        # Assert
        self.assertEqual(actual.shape, expected.shape)
        self.assertSequenceEqual(actual.ravel().tolist(), expected.ravel().tolist())

    def _invoke_sut__call__(self, entity, position_embeddings, tokens_array):
        # Initialise sut
        sut = PositionEmbedder(embeddings=position_embeddings)
        # Act
        actual = sut(tokens_array, entity)
        return actual

    def test__init_default_embedder(self):
        """
        Test the default initialiser
        """
        # Arrange
        pos_range = [0, 1, 2, 4, 6, 8]
        pos_dim = 2

        # Act
        sut = PositionEmbedder(pos_dim=pos_dim, pos_range=pos_range)
        print(sut.embeddings)

        # Assert
        self.assertEqual(sut.embeddings.shape, (len(pos_range), pos_dim))
