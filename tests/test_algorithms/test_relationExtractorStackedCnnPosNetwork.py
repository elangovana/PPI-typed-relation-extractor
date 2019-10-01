import os
from logging.config import fileConfig
from unittest import TestCase

import numpy as np
import torch

from algorithms.RelationExtractorStackedCnnPosNetwork import RelationExtractorStackedCnnPosNetwork


class TestRelationExtractorStackedCnnPosNetwork(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_forward(self):
        vocab_size = 10000
        batch_size = 30
        vector_dim = 5
        class_size = 2

        max_abstract_len = 50
        abstract = torch.LongTensor(batch_size, max_abstract_len).random_(0, vocab_size)

        max_itype_len = 1
        interaction_type = torch.LongTensor(batch_size, max_itype_len).random_(0, vocab_size)

        max_entype_len = 1
        entity = torch.LongTensor(batch_size, max_entype_len).random_(0, vocab_size)

        sut = RelationExtractorStackedCnnPosNetwork(class_size=class_size, embedding_dim=vector_dim,
                                                    feature_lengths=np.array(
                                                        [max_abstract_len, max_itype_len, max_entype_len]),
                                                    embed_vocab_size=vocab_size)

        # Act
        actual = sut((abstract, interaction_type, entity))

        # assert
        self.assertEqual(actual.shape, torch.Size([batch_size, class_size]))
