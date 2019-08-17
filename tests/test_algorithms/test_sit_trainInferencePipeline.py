import functools
import operator
import os
import tempfile
from io import StringIO
from logging.config import fileConfig
from unittest import TestCase

from torch.utils.data import DataLoader

from algorithms.CnnPosTrainInferenceBuilder import CnnPosTrainInferenceBuilder
from algorithms.Collator import Collator
from algorithms.PpiDataset import PPIDataset


class TestSitTrainInferencePipeline(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_call(self):
        # Arrange
        mock_dataset_train = self._get_mock_dataset()
        mock_dataset_val = self._get_mock_dataset()

        sut = self._get_sut_train_pipeline(mock_dataset_train)
        train_loader = DataLoader(mock_dataset_train, shuffle=True, collate_fn=Collator())
        val_loader = DataLoader(mock_dataset_val, shuffle=False, collate_fn=Collator())


        # Act

        actual = sut(train_loader, val_loader)

    def _get_sut_train_pipeline(self, mock_dataset, out_dir=tempfile.mkdtemp()):
        embedding = StringIO(
            "\n".join(["4 3", "hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9"]))
        factory = CnnPosTrainInferenceBuilder(dataset=mock_dataset, embedding_handle=embedding, embedding_dim=3,
                                              output_dir=out_dir, epochs=5)
        sut = factory.get_trainpipeline()
        return sut

    def _get_mock_dataset(self):
        # Arrange
        # Arrange
        train_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_train.json")
        mock_dataset = PPIDataset(train_file)
        return mock_dataset

    def test_predict(self):
        # Arrange
        mock_dataset_train = self._get_mock_dataset()
        mock_dataset_val = self._get_mock_dataset()
        out_dir = tempfile.mkdtemp()

        sut = self._get_sut_train_pipeline(mock_dataset_train, out_dir=out_dir)
        train_loader = DataLoader(mock_dataset_train, shuffle=True, collate_fn=Collator())
        val_loader = DataLoader(mock_dataset_val, shuffle=False, collate_fn=Collator())

        # get predictions
        # Todo: fix the return from sut.... it is not a batch of scores but flattened
        expected_scores, target, expected_predicted = sut(train_loader, val_loader)
        expected_predicted = expected_predicted.tolist()

        # Act
        predictor = sut.load(out_dir)
        predicted, confidence_scores = predictor(val_loader)

        # Assert
        flat_actual = functools.reduce(operator.iconcat, predicted, [])

        self.assertSequenceEqual(expected_predicted, flat_actual)
