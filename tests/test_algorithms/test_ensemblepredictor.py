from unittest import TestCase
from unittest.mock import MagicMock

from algorithms.ensemble_predictor import EnsemblePredictor


class TestEnsemblePredictor(TestCase):
    def test_predict(self):
        """
        Simple base case with single ensemble
        """
        # Arrange
        # output mock data
        predictions = [[1, 0]]
        # return a batch of results
        confidence_scores = [[[0.0715140849351883, 0.088773213326931], [0.0815140849351883, 0.078773213326931]]]
        # mock predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = predictions, confidence_scores

        mock_model = MagicMock
        models = mock_model

        sut = EnsemblePredictor(model_wrapper=mock_predictor)

        # Act
        actual_predictions, actual_confidence = sut.predict(models, input)

        # Assert
        self.assertSequenceEqual(predictions, actual_predictions)
        self.assertSequenceEqual(confidence_scores, actual_confidence)

    def test_predict_2(self):
        """
        Simple base case with 2 items in ensemble
        """
        # Arrange
        # output mock data
        predictions = [[1], [0]]
        confidence_scores = [[[0.0715140849351883, 0.088773213326931]], [[0.0815140849351883, 0.078773213326931]]]

        # mock predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = predictions, confidence_scores

        mock_model = MagicMock()
        # 2 models
        models = [mock_model, mock_model]

        sut = EnsemblePredictor(model_wrapper=mock_predictor)

        # Act
        actual_predictions, actual_confidence = sut.predict(models, input)

        # Assert
        self.assertSequenceEqual(predictions, actual_predictions)
        self.assertSequenceEqual(confidence_scores, actual_confidence)

    def test_predict_2_differnt_confidence(self):
        """
        Simple base case with 2 items in ensemble
        """
        # Arrange
        # output mock data
        predictions = [[1, 0]]
        confidence_scores_1 = [[[0.0, 1], [1.0, 0.0]]]
        confidence_scores_2 = [[[0.05, .95], [.80, 0.2]]]

        expected_confidence_scores = [[[0.05 / 2, 1.95 / 2], [1.8 / 2, 0.2 / 2]]]

        mock_model_1 = MagicMock()
        mock_model_2 = MagicMock()

        models = [mock_model_1, mock_model_2]

        # mock predictor
        mock_model_wrapper = MagicMock()

        def mock_model_wrapper_call(m, d, h):
            return (predictions, confidence_scores_1) if m == mock_model_1 else (predictions, confidence_scores_2)

        mock_model_wrapper.predict.side_effect = mock_model_wrapper_call

        # 2 models
        sut = EnsemblePredictor(model_wrapper=mock_model_wrapper)

        # Act
        actual_predictions, actual_confidence = sut.predict(models, input)

        # Assert
        self.assertSequenceEqual(predictions, actual_predictions)
        self.assertSequenceEqual(expected_confidence_scores, actual_confidence)
