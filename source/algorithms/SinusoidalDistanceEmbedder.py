import logging

import numpy as np
import torch

from algorithms.BaseDistanceEmbedder import BaseDistanceEmbedder


class SinusoidalDistanceEmbedder(BaseDistanceEmbedder):
    """
    Returns sinusoidal embedder
    """

    def __init__(self, max_pos=5, pos_dim=3):
        self.d_pos_vec = pos_dim
        self.max_pos = max_pos

    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self):
        pos_range = range(self.max_pos)
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.d_pos_vec) for i in range(self.d_pos_vec)]
            if pos != 0 else np.zeros(self.d_pos_vec) for pos in pos_range])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)
