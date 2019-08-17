import logging

import torch

"""
Reshape the label so that it matches the number of output predictions of the network
"""


class TransformLabelReshaper:

    def __init__(self, num_classes):
        self.num_classes = num_classes

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, data_loader):
        pass

    def transform(self, data_loader):
        # Check if iterable
        # Check if iterable
        try:
            int(data_loader)
            is_int = True
        except TypeError:
            is_int = False

        # Looks like single value
        if is_int:
            assert 0 <= data_loader < self.num_classes, "The value must be greater than equal to zero and less than {} ".format(
                self.num_classes)
            tensor = torch.tensor(data_loader)
            self.logger.info("Loading int {} to tensor {}".format(int(data_loader), tensor))
            return tensor

        batches = []
        for idx, b in enumerate(data_loader):
            b_x = b[0]
            b_y = b[1]
            dim = len(b_y.shape)
            assert dim == 1, "Expect a 1 dimensional tensor, but found  {} dimension".format(dim)
            encoded_y = torch.from_numpy(b_y)

            batches.append([b_x, encoded_y])
        return batches

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)
