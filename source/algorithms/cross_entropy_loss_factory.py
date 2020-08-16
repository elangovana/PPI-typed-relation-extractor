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

from torch.nn import CrossEntropyLoss

from algorithms.loss_factory_base import LossFactoryBase


class CrossEntropyLossFactory(LossFactoryBase):
    """
    Returns the CrossEntropyLoss
    """

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def get(self, **kwargs):
        loss = CrossEntropyLoss()

        return loss
