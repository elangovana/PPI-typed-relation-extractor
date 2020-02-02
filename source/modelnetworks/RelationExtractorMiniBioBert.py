import logging

import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from torch import nn


class RelationExtractorMiniBioBert(nn.Module):

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __init__(self, model_dir, num_classes, seed=None, num_layers=5):
        super().__init__()

        if seed is None:
            seed = torch.initial_seed() & ((1 << 63) - 1)
        self.logger.info("Using seed {}".format(seed))
        torch.manual_seed(seed)

        model_full = BertForSequenceClassification.from_pretrained(model_dir,
                                                                   num_labels=num_classes)
        # Modify bert to use smaller number of num_layers
        model_full.bert.encoder.layer = model_full.bert.encoder.layer[0: num_layers]

        self.model = model_full
        print(self.model)

    def forward(self, input):
        return self.model(input)
