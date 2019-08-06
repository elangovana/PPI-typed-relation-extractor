import os
import tempfile
from io import StringIO
from logging.config import fileConfig
from unittest import TestCase
from unittest.mock import MagicMock

from torch.utils.data import DataLoader

from algorithms.CnnPosTrainInferenceBuilder import CnnPosTrainInferenceBuilder


class TestSitTrainInferencePipeline(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_call(self):
        # Arrange
        out_dir = tempfile.mkdtemp()
        embedding = StringIO(
            "\n".join(["hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9"]))

        # Arrange
        # Arrange
        mock_dataset = MagicMock()
        mock_dataset.data = [[["This is sample text", "entity1", "entity2", "phosphorylation"], "yes"],
                             [["This is sample text2", "entity1", "entity2", "phosphorylation"], "no"]]
        mock_dataset.feature_lens = [10, 1, 1, 1]
        mock_dataset.class_size = 2
        mock_dataset.postive_label = "yes"

        mock_dataset.__len__.return_value = len(mock_dataset.data)
        mock_dataset.__getitem__.side_effect = lambda i: (mock_dataset.data[i][0], mock_dataset.data[i][1])

        val_data = [["This is hat", "entity1", "entity2"],
                    ["this is a cat not a mat", "mat protein", "cat protein"]]

        cols = ['abstract', 'entity1', 'entity2']

        factory = CnnPosTrainInferenceBuilder(dataset=mock_dataset, embedding_handle=embedding, embedding_dim=3,
                                              output_dir=out_dir)

        sut = factory.get_trainpipeline()

        # Act
        actual = sut(DataLoader(mock_dataset), DataLoader(mock_dataset))

    # def test_call_cnn(self):
    #     # Arrange
    #     out_dir = tempfile.mkdtemp()
    #     embedding = StringIO(
    #         "\n".join(["hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9"]))
    #     sut = TrainInferencePipeline(class_size=2, embedding_handle=embedding, embedding_dim=3, ngram=1,
    #                                  output_dir=out_dir, pos_label=True)
    #     sut.model_network = RelationExtractorCnnNetwork
    #     train_df = [["This is good", "entity1", "entity2"],
    #                 ["this is a cat not a hat", "mat protein", "cat protein"]]
    #
    #     val_data = [["This is good", "entity1", "entity2"],
    #                 ["this is a cat not a mat", "mat protein", "cat protein"]]
    #
    #     labels = [True, False]
    #     cols = ['abstract', 'entity1', 'entity2']
    #     train_df = pd.DataFrame(train_df, columns=cols)
    #     val_df = pd.DataFrame(val_data, columns=cols)
    #
    #     # Act
    #     actual = sut(train_df, labels, val_df, labels)
    #
    # def test_call_cnnpos(self):
    #     # Arrange
    #     out_dir = tempfile.mkdtemp()
    #     embedding = StringIO(
    #         "\n".join(["hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9"]))
    #     sut = TrainInferencePipeline(class_size=2, embedding_handle=embedding, embedding_dim=3, ngram=1,
    #                                  output_dir=out_dir, pos_label=True,
    #                                  class_weights_dict={True: 2, False: 1})
    #     sut.model_network = RelationExtractorCnnPosNetwork
    #     train_df = [["This is good", "entity1", "entity2"],
    #                 ["this is a cat not a hat", "mat protein", "cat protein"]]
    #
    #     val_data = [["This is good", "entity1", "entity2"],
    #                 ["this is a cat not a mat", "mat protein", "cat protein"]]
    #
    #     labels = [True, False]
    #     cols = ['abstract', 'entity1', 'entity2']
    #     train_df = pd.DataFrame(train_df, columns=cols)
    #     val_df = pd.DataFrame(val_data, columns=cols)
    #
    #     # Act
    #     actual = sut(train_df, labels, val_df, labels)

    # def test_predict(self):
    #     # Arrange
    #     out_dir = tempfile.mkdtemp()
    #
    #     embedding = StringIO(
    #         "\n".join(["hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9"]))
    #     pos_label = True
    #     sut = TrainInferencePipeline()
    #
    #     train_df = [["This is good", "entity1", "entity2"],
    #                 ["this is a cat not a hat", "mat protein", "cat protein"]]
    #
    #     val_data = [["This is hat", "entity1", "entity2"],
    #                 ["this is a cat not a mat jack", "mat protein", "cat protein"],
    #                 ["this is block four", "mat protein", "cat protein"],
    #                 ["this is white four five", "mat protein", "cat protein"]
    #                 ]
    #
    #     labels = [True, False]
    #     val_label = [True, False, True, False]
    #     cols = ['abstract', 'entity1', 'entity2']
    #     train_df = pd.DataFrame(train_df, columns=cols)
    #     val_df = pd.DataFrame(val_data, columns=cols)
    #
    #     model, expected_scores, target, expected_predicted = sut(train_df, labels, val_df, val_label)
    #
    #     # Act
    #     predictor = TrainInferencePipeline.load(out_dir)
    #     actual, confidence_scores = predictor(val_df)
    #
    #     # Act
    #     scorer = ResultScorerF1()
    #     predictions = scorer(y_pred=actual, y_actual=val_label, pos_label=pos_label)
    #
    #     self.assertEqual(expected_scores, predictions)
