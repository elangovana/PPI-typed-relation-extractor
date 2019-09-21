from unittest import TestCase
from unittest.mock import MagicMock

import torch
from torch.utils.data import DataLoader

from algorithms.transform_protein_mask import TransformProteinMask


class TestTransformProteinMask(TestCase):
    def test_transform(self):
        # Arrange

        # Mock data set
        mock_dataset = MagicMock()
        mock_dataset.data = [[["This is sample entity1 entity1", "entity1", "entity2", "phosphorylation"], "yes"],
                             [["This is sample text2", "entity1", "entity2", "phosphorylation"], "no"]]
        mock_dataset.feature_lens = [10, 1, 1, 1]
        mock_dataset.class_size = 2
        mock_dataset.positive_label = "yes"
        mock_dataset.__len__.return_value = len(mock_dataset.data)
        mock_dataset.__getitem__.side_effect = lambda i: (mock_dataset.data[i][0], mock_dataset.data[i][1])

        expected = [
            [[["This is sample PROTEIN_1 PROTEIN_1"], ["PROTEIN_1"], ("entity2",), ("phosphorylation",)], ("yes",)],
            [[["This is sample text2"], ["PROTEIN_1"], ("entity2",), ("phosphorylation",)], ("no",)]]

        sut = TransformProteinMask(entity_column_index=1, text_column_index=0, mask="PROTEIN_1")

        # Act
        actual = sut.fit_transform(DataLoader(mock_dataset))

        # Assert
        self.assertSequenceEqual(expected, actual)

    def test_transform_with_offset(self):
        # Arrange

        # Mock data set
        mock_dataset = MagicMock()
        mock_dataset.data = [
            [["entity2 This is sample entity1 entity1 ", "entity1", 23, "entity2", 0, "phosphorylation"], "yes"],
            ]
        mock_dataset.feature_lens = [10, 1, 1, 1]
        mock_dataset.class_size = 2
        mock_dataset.positive_label = "yes"
        mock_dataset.__len__.return_value = len(mock_dataset.data)
        mock_dataset.__getitem__.side_effect = lambda i: (mock_dataset.data[i][0], mock_dataset.data[i][1])

        expected = [
            [[["entity2 This is sample PROTEIN_1 entity1 "], ["PROTEIN_1"], torch.tensor([23]), ("entity2",), torch.tensor([0]),("phosphorylation",)], ("yes",)],
            ]

        sut = TransformProteinMask(entity_column_index=1, text_column_index=0, mask="PROTEIN_1", entity_offset_index=2)

        # Act
        actual = sut.fit_transform(DataLoader(mock_dataset))

        # Assert
        self.assertSequenceEqual(expected, actual)
