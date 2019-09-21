from unittest import TestCase
from unittest.mock import MagicMock

from torch.utils.data import DataLoader

from algorithms.transform_sentence_tokeniser import TransformSentenceTokenisor


class TestTransformSentenceTokenisor(TestCase):
    def test_transform(self):
        # Arrange

        # Mock data set
        mock_dataset = MagicMock()
        mock_dataset.data = [
            [["This is sample entity1. But ths is a new sentence", "entity1", "entity2", "phosphorylation"], "yes"],
            [["This is sample text2", "entity1", "entity2", "phosphorylation"], "no"]]
        mock_dataset.feature_lens = [10, 1, 1, 1]
        mock_dataset.class_size = 2
        mock_dataset.positive_label = "yes"
        mock_dataset.__len__.return_value = len(mock_dataset.data)
        mock_dataset.__getitem__.side_effect = lambda i: (mock_dataset.data[i][0], mock_dataset.data[i][1])

        expected = [[[["This is sample entity1 <EOS> But ths is a new sentence"], ("entity1",), ("entity2",),
                      ("phosphorylation",)], ("yes",)],
                    [[["This is sample text2"], ("entity1",), ("entity2",), ("phosphorylation",)], ("no",)]]

        sut = TransformSentenceTokenisor(text_column_index=0, eos_token="<EOS>")

        # Act
        actual = sut.fit_transform(DataLoader(mock_dataset))

        # Assert
        self.assertSequenceEqual(expected, actual)
