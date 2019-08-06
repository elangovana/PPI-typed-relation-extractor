import logging
import os

import torch


class ModelSnapshotCallback:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, model, current_score, best_score_so_far, output_dir, prefix="best_snaphsot"):
        # found a model with better validation set accuracy

        snapshot_prefix = os.path.join(output_dir, prefix)
        snapshot_path = snapshot_prefix + 'model.pt'

        if current_score > best_score_so_far:
            self.logger.info("Snappshotting model becayse the current {} is greater than previous best {}".format(
                current_score, best_score_so_far))
            # save model, delete previous 'best_snapshot' files
            torch.save(model, snapshot_path)

            best_score_so_far = current_score

        return best_score_so_far
