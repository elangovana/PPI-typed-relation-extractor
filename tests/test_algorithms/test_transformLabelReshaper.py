from unittest import TestCase

import torch

from algorithms.transform_label_rehaper import TransformLabelReshaper


class TestTransformLabelReshaper(TestCase):
    def test_transform_single_number(self):
        # Arrange
        input = 2
        sut = TransformLabelReshaper(num_classes=3)
        expected = torch.Tensor([input]).long()

        # Act
        actual = sut.fit_transform(input)

        # Assert
        self.assertTrue(expected.equal(actual), "Expected {}, but found {}".format(expected, actual))
