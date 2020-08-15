from unittest import TestCase

import torch
from torch.nn import CrossEntropyLoss

from algorithms.top_k_cross_entropy_loss import TopKCrossEntropyLoss


class TestTopKCrossEntropyLoss(TestCase):
    def test_forward_one_item(self):
        # Arrange
        k = 1
        predicted = torch.tensor([[0.1, 0.9]])
        target = torch.tensor([0])
        expected_loss = CrossEntropyLoss()(predicted, target)

        sut = TopKCrossEntropyLoss(k)

        # Act
        actual = sut.forward(predicted, target)

        # Assert
        self.assertEqual(round(expected_loss.item(), 2), round(actual.item(), 2))

    def test_forward(self):
        # Arrange
        k = 2
        predicted = torch.tensor([[0.5, .5], [1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([0, 1, 0])
        expected_loss = CrossEntropyLoss()(predicted[torch.tensor([1, 2])], target[torch.tensor([1, 2])])

        sut = TopKCrossEntropyLoss(k)

        # Act
        actual = sut.forward(predicted, target)

        # Assert
        self.assertEqual(round(expected_loss.item(), 2), round(actual.item(), 2))
