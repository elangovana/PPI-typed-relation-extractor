from io import StringIO
from unittest import TestCase
from unittest.mock import MagicMock

from torch.utils.data import DataLoader

from algorithms.DataPipeline import DataPipeline
from algorithms.PretrainedEmbedderLoader import PretrainedEmbedderLoader


class TestSitDataPipeline(TestCase):

    def test_fit(self):
        # Arrange
        embedding = StringIO(
            "\n".join(["hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9"]))

        mock_dataset = MagicMock()
        mock_dataset.data = [[["This is sample text", "entity1", "entity2", "phosphorylation"], "yes"],
                             [["This is sample text2", "entity1", "entity2", "phosphorylation"], "no"]]
        mock_dataset.feature_lens = [10, 1, 1, 1]
        mock_dataset.class_size = 2
        mock_dataset.positive_label = "yes"
        mock_dataset.__len__.return_value = len(mock_dataset.data)
        mock_dataset.__getitem__.side_effect = lambda i: (mock_dataset.data[i][0], mock_dataset.data[i][1])

        mock_embedder_loader = PretrainedEmbedderLoader()
        sut = DataPipeline(text_to_index=MagicMock())

        # Act
        sut.fit_transform(DataLoader(mock_dataset))
