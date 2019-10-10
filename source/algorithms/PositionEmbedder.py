import logging

import numpy as np

from algorithms.RawDistanceEmbedder import RawDistanceEmbedder


class PositionEmbedder():
    """
    Embeds position vectors  with the distance to the nearest entity
    """

    def __init__(self, embeddings=None, pad_token_id=None):
        self.pad_token_id = pad_token_id
        self.embeddings = embeddings if embeddings is not None else RawDistanceEmbedder()()

    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, tokens_array, entity):
        entity_positions = np.asarray([p for p, t in enumerate(tokens_array) if t == entity])
        max_distance = self.embeddings.shape[0] - 1

        if len(entity_positions) == 0:
            logging.debug(
                "The entity {} was not found in the tokens array. Hence returning max distance for all tokens".format(
                    entity))
            logging.debug("The entity {} was not found in the tokens array {}".format(entity, tokens_array))
            # returning max distance
            token_distance = [max_distance] * len(tokens_array)

        else:
            # Get shortest distance between entities and tokens
            # Also make sure all the tokens are limited to the max distance
            get_distance = lambda p: min(max_distance, np.min(np.abs(entity_positions - p)))

            token_distance = [get_distance(p) for p, t in
                              enumerate(tokens_array)]

        result = self.embeddings[token_distance]

        # Replace all pad token positions to zero
        if self.pad_token_id is not None:
            pad_token_indices = [p for p, t in
                                 enumerate(tokens_array) if t == self.pad_token_id]

            result[pad_token_indices] = 0

        return result
