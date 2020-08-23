# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
import logging

import torch
from torch import nn


class TopKCrossEntropyLoss(nn.Module):
    """
    Computes the Cross entropy loss for the top k most difficult samples
    """

    def __init__(self, k):
        super().__init__()
        self.k = k
        self._loss_func = nn.CrossEntropyLoss(reduction='none')

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def forward(self, predicted, target):
        # make sure k is within the length of the target shape
        k = min(self.k, target.shape[0])

        loss_per_item = self._loss_func(predicted, target)

        # Obtain only the top k hard samples
        top_k_loss = torch.mean(torch.topk(loss_per_item, k=k)[0])
        self._logger.debug("Total loss {} vs topk loss {}".format(torch.mean(loss_per_item), top_k_loss))


        return top_k_loss
