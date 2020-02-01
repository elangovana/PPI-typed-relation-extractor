import os
import tempfile
from unittest import TestCase

from algorithms.dataset_factory import DatasetFactory
from trainpipelinesbuilders.BertTrainInferenceBuilder import BertTrainInferenceBuilder


class ITTestBertTrainInferencePipeline(TestCase):

    def test_call_ppidataset(self):
        # Arrange
        mock_dataset_train, scorer = self._get_ppidataset()
        mock_dataset_val, _ = self._get_ppidataset()

        sut = self._get_sut_train_pipeline(mock_dataset_train, scorer=scorer)

        # Act
        actual = sut(mock_dataset_train, mock_dataset_val)

    def test_call_aimeddataset(self):
        # Arrange
        mock_dataset_train, scorer = self._get_aimeddataset()
        mock_dataset_val, _ = self._get_aimeddataset()

        sut = self._get_sut_train_pipeline(mock_dataset_train, scorer=scorer)


        # Act
        actual = sut(mock_dataset_train, mock_dataset_val)

    def test_call_interactiondataset(self):
        # Arrange
        mock_dataset_train, scorer = self._get_interactiondataset()
        mock_dataset_val, _ = self._get_interactiondataset()

        sut = self._get_sut_train_pipeline(mock_dataset_train, scorer=scorer)

        # Act
        actual = sut(mock_dataset_train, mock_dataset_val)

    def test_call_aimeddataset_bertlstm(self):
        # Arrange
        mock_dataset_train, scorer = self._get_aimeddataset()
        mock_dataset_val, _ = self._get_aimeddataset()

        sut = self._get_sut_train_pipeline(mock_dataset_train,
                                           network_factory_name="RelationExtractorBertBiLstmNetworkNoPosFactory")

        # Act
        actual = sut(mock_dataset_train, mock_dataset_val)

    def _get_sut_train_pipeline(self, mock_dataset, out_dir=tempfile.mkdtemp(), epochs=1,
                                network_factory_name="RelationExtractorBioBertFactory", scorer=None):
        base_model_dir = os.path.join(os.path.dirname(__file__), "..", "temp", "biobert")

        factory = BertTrainInferenceBuilder(dataset=mock_dataset,
                                            output_dir=out_dir, model_dir=out_dir, epochs=epochs,
                                            network_factory_name=network_factory_name,
                                            extra_args={"pretrained_biobert_dir": base_model_dir},
                                            results_scorer=scorer)
        sut = factory.get_trainpipeline()
        return sut

    def _get_ppidataset(self):
        # Arrange
        train_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_train.json")
        factory = DatasetFactory().get_datasetfactory("PpiDatasetFactory")
        dataset = factory.get_dataset(train_file)
        scorer = factory.get_metric_factory().get()
        return dataset, scorer

    def _get_aimeddataset(self):
        train_file = os.path.join(os.path.dirname(__file__), "..", "data", "Aimedsample.json")
        factory = DatasetFactory().get_datasetfactory("PpiAimedDatasetFactory")
        dataset = factory.get_dataset(train_file)
        scorer = factory.get_metric_factory().get()

        return dataset, scorer

    def _get_interactiondataset(self):
        train_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_classification.json")
        factory = DatasetFactory().get_datasetfactory("InteractionDatasetFactory")
        dataset = factory.get_dataset(train_file)
        scorer = factory.get_metric_factory().get()

        return dataset, scorer

    def test_predict(self):
        # Arrange
        mock_dataset_train, scorer = self._get_ppidataset()
        mock_dataset_val, _ = self._get_ppidataset()
        out_dir = tempfile.mkdtemp()

        sut = self._get_sut_train_pipeline(mock_dataset_train, out_dir=out_dir, epochs=4, scorer=scorer)

        # get predictions
        # Todo: fix the return from sut.... it is not a batch of scores but flattened
        expected_scores, target, expected_predicted = sut(mock_dataset_train, mock_dataset_val)
        expected_predicted = expected_predicted.tolist()

        # Act
        predictor = sut.load(out_dir)
        predicted, confidence_scores = predictor(mock_dataset_val)

        # Assert
        self.assertSequenceEqual(expected_predicted, predicted.tolist())
