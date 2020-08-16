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

from algorithms.loss_factory_base import LossFactoryBase
from algorithms.top_k_cross_entropy_loss import TopKCrossEntropyLoss


class TopKCrossEntropyLossFactory(LossFactoryBase):
    """
    Returns the TopKCrossEntropyLoss
    """

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self._logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

    def get(self, **kwargs):
        k = int(self._get_value(kwargs, "top_k_loss", "32"))

        loss = TopKCrossEntropyLoss(k=k)

        return loss
