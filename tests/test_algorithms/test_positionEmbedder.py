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

    def _invoke_sut__call__(self, entity, position_embeddings, tokens_array):
        # Initialise sut
        sut = PositionEmbedder(position_embeddings)
        # Act
        actual = sut(tokens_array, entity)
        return actual
