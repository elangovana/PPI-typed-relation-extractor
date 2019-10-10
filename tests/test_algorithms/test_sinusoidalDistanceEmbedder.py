from unittest import TestCase

from algorithms.SinusoidalDistanceEmbedder import SinusoidalDistanceEmbedder


class TestSinusoidalDistanceEmbedder(TestCase):
    def test__call__(self):
        """
        Test the default initialiser
        """
        # Arrange
        max_pos = 5
        pos_dim = 3
        sut = SinusoidalDistanceEmbedder(max_pos=max_pos, pos_dim=pos_dim)

        # Act
        actual = sut()

        # Assert
        self.assertEqual(actual.shape, (max_pos, pos_dim))
