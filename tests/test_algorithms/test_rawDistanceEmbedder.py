from unittest import TestCase

import torch

from algorithms.RawDistanceEmbedder import RawDistanceEmbedder


class TestRawDistanceEmbedder(TestCase):
    def test__call__(self):
        # Arrange
        sut = RawDistanceEmbedder(max_pos=5)
        expected = torch.tensor([[0, 1, 2, 3, 4, 5]]).float()

        # Act
        actual = sut()

        # Assert
        self.assertTrue(torch.all(expected.eq(actual)), "The expected and actual tensors do not match")
