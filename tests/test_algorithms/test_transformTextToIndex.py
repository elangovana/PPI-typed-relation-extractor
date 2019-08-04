from unittest import TestCase
from unittest.mock import MagicMock

from torch.utils.data import DataLoader

from algorithms.transform_text_index import TransformTextToIndex


class TestTransformTextToIndex(TestCase):

    def test_transform(self):
        mock_dataset = MagicMock()
        mock_dataset.data = [[["This is sample text", "entity1", "entity2", "phosphorylation"], ["yes"]],
                             [["This is sample text2", "entity1", "entity2", "phosphorylation"], ["no"]]]
        max_feature_lens = [10, 1, 1, 1]
        # Unique words + pad character ( ignore labels)
        expected_unique_item_no = 9

        mock_dataset.__len__.return_value = len(mock_dataset.data)

        mock_dataset.__getitem__.side_effect = lambda i: (mock_dataset.data[i][0], mock_dataset.data[i][1])

        sut = TransformTextToIndex(max_feature_lens)

        data_loader = DataLoader(mock_dataset, batch_size=2)

        # Act
        actual = list(sut.fit_transform(data_loader, None))

        # Assert the max feature length matchs
        unique_items = set()
        for b in actual:
            for ci, c in enumerate(b):
                for r in c:
                    unique_items = unique_items.union(r)
                    feature_len = max_feature_lens[ci]
                    self.assertEqual(feature_len, len(r),
                                     "The feature length for column {} should match the max_feature_length".format(
                                         feature_len))

        # Assert
        self.assertEqual(expected_unique_item_no, len(unique_items),
                         "The number of unique words doesnt match to unique indexes including padding{}".format(
                             unique_items))