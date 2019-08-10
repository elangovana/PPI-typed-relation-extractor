from unittest import TestCase

import torch

from algorithms.RelationExtractorLinearNetwork import RelationExtractorLinearNetwork


class TestRelationExtractorLinearNetwork(TestCase):
    def test_forward(self):
        vocab_size = 10000
        batch_size = 10
        vector_dim = 200
        class_size = 2

        max_abstract_len = 10
        abstract = torch.LongTensor(batch_size, max_abstract_len).random_(0, vocab_size)

        max_itype_len = 1
        interaction_type = torch.LongTensor(batch_size, max_itype_len).random_(0, vocab_size)


        sut = RelationExtractorLinearNetwork(class_size=class_size, embedding_dim=vector_dim,
                                             pretrained_weights_or_embed_vocab_size=vocab_size,
                                             feature_lengths=[max_abstract_len, max_itype_len])

        # Act
        actual = sut([abstract, interaction_type])

        # assert
        self.assertEqual(actual.shape, torch.Size([batch_size, class_size]))
