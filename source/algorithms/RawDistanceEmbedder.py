import logging

import torch

from algorithms.BaseDistanceEmbedder import BaseDistanceEmbedder


class RawDistanceEmbedder(BaseDistanceEmbedder):
    """
    Returns the distance as is for embedding
    """

    def __init__(self, max_pos=5):
        self.max_pos = max_pos

    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self):
        position_enc = torch.tensor([[i for i in range(self.max_pos + 1)]]).float()
        return position_enc
