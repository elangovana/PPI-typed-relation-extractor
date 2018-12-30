import logging

import numpy

"""
Embeds position vectors  with the distance to the nearest entity
"""


class PositionEmbedder:

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, tokens_array, entity):
        entity_positions = numpy.asarray([p for p, t in enumerate(tokens_array) if t == entity])
        max_distance = self.embeddings.shape[0] - 1
        if len(entity_positions) == 0:
            logging.warning("The entity {} was not found in the tokens array {}".format(entity, tokens_array))
            return []

        # Get shortest distance between entities and tokens
        # Also make sure all the tokens are limited to the max distance
        token_distance = [min(max_distance, numpy.min(numpy.abs(entity_positions - p))) for p, t in
                          enumerate(tokens_array)]

        return self.embeddings[token_distance]
