import logging
import os

import torch


class ModelSnapshotCallback:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, model, output_dir, prefix="best_snaphsot"):
        # found a model with better validation set accuracy

        snapshot_prefix = os.path.join(output_dir, prefix)
        snapshot_path = snapshot_prefix + 'model.pt'

        self.logger.info("Snappshotting model to {}".format(snapshot_path))
        # save model, delete previous 'best_snapshot' files

        torch.save(model, snapshot_path)
