import os
import tempfile
from io import StringIO
from logging.config import fileConfig
from unittest import TestCase

import pandas as pd

from algorithms.RelationExtractionLinearDropoutWordFactory import RelationExtractorLinearNetworkDropoutWordFactory
from algorithms.result_scorer import ResultScorer


class TestSitRelationExtractionLinearFactoryWordDropout(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_call(self):
        # Arrange
        out_dir = tempfile.mkdtemp()
        embedding = StringIO(
            "\n".join(["hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9"]))
        sut = RelationExtractorLinearNetworkDropoutWordFactory(class_size=2, embedding_handle=embedding,
                                                               embedding_dim=3, ngram=1,
                                                               output_dir=out_dir, pos_label="1", min_vocab_frequency=1)

        train_df = [["This is good", "entity1", "entity2"],
                    ["this is a cat not a hat", "mat protein", "cat protein"]]

        val_data = [["This is hat", "entity1", "entity2"],
                    ["this is a cat not a mat", "mat protein", "cat protein"]]

        labels = ["1", "0"]
        cols = ['abstract', 'entity1', 'entity2']
        train_df = pd.DataFrame(train_df, columns=cols)
        val_df = pd.DataFrame(val_data, columns=cols)

        # Act
        actual = sut(train_df, labels, val_df, labels)

    def test_predict(self):
        # Arrange
        out_dir = tempfile.mkdtemp()
        embedding = StringIO(
            "\n".join(
                ["hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9", "block .1 .2 .3",
                 "mode .2 .4 .6"]))
        pos_label = 1
        sut = RelationExtractorLinearNetworkDropoutWordFactory(class_size=2, embedding_handle=embedding,
                                                               embedding_dim=3, ngram=1,
                                                               output_dir=out_dir, pos_label=pos_label,
                                                               min_vocab_frequency=1)

        train_df = [["This is good", "entity1", "entity2"],
                    ["this is a cat not a hat block mode", "mat protein", "cat protein"]]

        val_data = [["This is hat", "entity1", "entity2"],
                    ["this is a cat not a mat", "mat protein", "cat protein"],
                    ["this is block not ", "mat protein", "cat protein"],
                    ["this is white mnot mode", "mat protein", "cat protein"]
                    ]

        labels = [1, 0]
        val_label = [1, 1, 1, 0]
        cols = ['abstract', 'entity1', 'entity2']
        train_df = pd.DataFrame(train_df, columns=cols)
        val_df = pd.DataFrame(val_data, columns=cols)

        model, expected_scores, expected_actual, expected_predicted = sut(train_df, labels, val_df, val_label)

        # Act
        predictor = RelationExtractorLinearNetworkDropoutWordFactory.load(out_dir)
        actual, confidence_scores = predictor(val_df)

        # Assert
        scorer = ResultScorer()
        actual_scores = scorer(y_pred=actual, y_actual=val_label, pos_label=pos_label)
        self.assertSequenceEqual(expected_scores, actual_scores)
