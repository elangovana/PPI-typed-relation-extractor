import logging

import numpy as np
import torch

"""
Embeds position vectors  with the distance to the nearest entity
"""


class PositionEmbedder:

    def __init__(self, embeddings=None, pos_dim=3, pos_range=None):
        pos_range = pos_range if pos_range is not None else range(0, 10)
        self.embeddings = embeddings if embeddings is not None else self.get_embedder(pos_dim, pos_range)

    def get_embedder(self, d_pos_vec, pos_range):
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)]
            if pos != 0 else np.zeros(d_pos_vec) for pos in pos_range])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, tokens_array, entity):
        entity_positions = np.asarray([p for p, t in enumerate(tokens_array) if t == entity])
        max_distance = self.embeddings.shape[0] - 1
        if len(entity_positions) == 0:
            logging.warning("The entity {} was not found in the tokens array {}".format(entity, tokens_array))
            return []

        # Get shortest distance between entities and tokens
        # Also make sure all the tokens are limited to the max distance
        token_distance = [min(max_distance, np.min(np.abs(entity_positions - p))) for p, t in
                          enumerate(tokens_array)]

        return self.embeddings[token_distance]
