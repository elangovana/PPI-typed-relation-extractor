import os
from logging.config import fileConfig
from unittest import TestCase

import numpy as np
import torch

from modelnetworks.RelationExtractorBiLstmNetwork import RelationExtractorBiLstmNetwork


class TestRelationExtractorBiLstmNetwork(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_forward(self):
        vocab_size = 10000
        batch_size = 30
        vector_dim = 5
        class_size = 2

        max_abstract_len = 10
        abstract = torch.LongTensor(batch_size, max_abstract_len).random_(0, vocab_size)

        entity_markers = [1, 2]

        sut = RelationExtractorBiLstmNetwork(class_size=class_size, embedding_dim=vector_dim, feature_lengths=np.array(
            [max_abstract_len]), embed_vocab_size=vocab_size, entity_markers=entity_markers)

        # Act
        actual = sut((abstract,))

        # assert
        self.assertEqual(torch.Size([batch_size, class_size]), actual.shape)
