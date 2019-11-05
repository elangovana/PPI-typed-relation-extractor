import logging

import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from torch import nn


class RelationExtractorBioBert(nn.Module):

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __init__(self, model_dir, num_classes, seed=None):
        super().__init__()

        if seed is None:
            seed = torch.initial_seed() & ((1 << 63) - 1)
        self.logger.info("Using seed {}".format(seed))
        torch.manual_seed(seed)

        self.model = BertForSequenceClassification.from_pretrained(model_dir,
                                                                   num_labels=num_classes)

        print(self.model)

    def forward(self, input):
        return self.model(input)
