import os
from logging.config import fileConfig
from unittest import TestCase

import numpy as np
import torch

from algorithms.RelationExtractorCnnPosNetwork import RelationExtractorCnnPosNetwork


class TestRelationExtractorCnnPosNetwork(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_forward(self):
        vocab_size = 10000
        batch_size = 30
        vector_dim = 5
        class_size = 2

        max_abstract_len = 10
        abstract = torch.LongTensor(max_abstract_len, batch_size).random_(0, vocab_size)

        max_itype_len = 1
        interaction_type = torch.LongTensor(max_itype_len, batch_size).random_(0, vocab_size)

        max_entype_len = 1
        entity = torch.LongTensor(max_itype_len, batch_size).random_(0, vocab_size)

        sut = RelationExtractorCnnPosNetwork(class_size=class_size, embedding_dim=vector_dim,
                                             pretrained_weights_or_embed_vocab_size=vocab_size,
                                             feature_lengths=np.array(
                                                 [max_abstract_len, max_itype_len, max_entype_len]))

        # Act
        actual = sut((abstract, interaction_type, entity))

        # assert
        self.assertEqual(actual.shape, torch.Size([batch_size, class_size]))
