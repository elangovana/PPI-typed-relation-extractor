from unittest import TestCase
from unittest.mock import MagicMock

from torch.utils.data import DataLoader

from algorithms.transform_text_index import TransformTextToIndex


class TestTransformTextToIndex(TestCase):

    def test_transform_no_vocab(self):
        mock_dataset = MagicMock()
        initial_vocab_dict = None  # ["random", "initial"]
        mock_dataset.data = [[["This is sample text", "entity1", "entity2", "phosphorylation"], ["yes"]],
                             [["Completey random text2", "entity11", "entity12", "phosphorylation1"], ["no"]]]
        max_feature_lens = [10, 1, 1, 1]
        # Unique words + pad character ( ignore labels)
        expected_unique_item_no = 13 + 1

        mock_dataset.__len__.return_value = len(mock_dataset.data)

        mock_dataset.__getitem__.side_effect = lambda i: (mock_dataset.data[i][0], mock_dataset.data[i][1])

        sut = TransformTextToIndex(max_feature_lens, vocab_dict=initial_vocab_dict)

        data_loader = DataLoader(mock_dataset, batch_size=2)

        # Act
        actual = list(sut.fit_transform(data_loader))

        # Assert the max feature length matchs
        unique_items = set()
        for b, y in actual:
            for ci, c_tensor in enumerate(b):
                c = c_tensor.tolist()
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

    def test_transform_with_vocab(self):
        mock_dataset = MagicMock()
        initial_vocab_dict = {"random": 0, "initial": 1}
        mock_dataset.data = [[["This is sample text", "entity1", "entity2", "phosphorylation"], ["yes"]],
                             [["Minoritt word random unknown", "entity1", "entity2", "phosphorylation"], ["no"]]]
        max_feature_lens = [10, 1, 1, 1]
        # Unique words + pad character ( ignore labels)
        expected_unique_item_no = 12

        mock_dataset.__len__.return_value = len(mock_dataset.data)

        mock_dataset.__getitem__.side_effect = lambda i: (mock_dataset.data[i][0], mock_dataset.data[i][1])

        sut = TransformTextToIndex(max_feature_lens, vocab_dict=initial_vocab_dict, use_dataset_vocab=True)

        data_loader = DataLoader(mock_dataset, batch_size=2)

        # Act
        vocab_dict = sut.construct_vocab_dict(data_loader)

        sut.vocab_dict = vocab_dict

        actual = list(sut.fit_transform(data_loader))

        # Assert the max feature length matchs
        unique_items = set()
        for b, y in actual:
            for ci, c_tensor in enumerate(b):
                c = c_tensor.tolist()
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

    def test_transform_pad(self):
        """
        Test case to make ensure that pad index is zero
        """
        mock_dataset = MagicMock()
        initial_vocab_dict = {"random": 0, "initial": 1}
        mock_dataset.data = [[["This is sample text", "entity1", "entity2", "phosphorylation"], ["yes"]],
                             [["This is sample text2", "entity1", "entity2", "phosphorylation"], ["no"]]]
        max_feature_lens = [10, 1, 1, 1]
        # Unique words + pad character ( ignore labels)
        expected_unique_item_no = 9

        mock_dataset.__len__.return_value = len(mock_dataset.data)

        mock_dataset.__getitem__.side_effect = lambda i: (mock_dataset.data[i][0], mock_dataset.data[i][1])

        sut = TransformTextToIndex(max_feature_lens, vocab_dict=initial_vocab_dict, use_dataset_vocab=True)

        data_loader = DataLoader(mock_dataset, batch_size=2)

        # Act
        vocab_dict = sut.construct_vocab_dict(data_loader)

        # Assert the max feature length matchs
        self.assertEqual(vocab_dict[sut.pad_token()], 0, "Index of pas token {} must be zero".format(sut.pad_token()))
