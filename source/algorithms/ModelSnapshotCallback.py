import logging
import os

import torch

from algorithms.result_scorer import ResultScorer, score_type_f1


class ModelSnapshotCallback:

    def __init__(self):
        self.score_evaluator = None

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def score_evaluator(self):
        self.__score_evaluator__ = self.__score_evaluator__ or ResultScorer()
        return self.__score_evaluator__

    @score_evaluator.setter
    def score_evaluator(self, value):
        self.__score_evaluator__ = value

    def __call__(self, model, val_iter, best_accuracy, output_dir, pos_label, prefix="best_snaphsot",
                 metric=score_type_f1):
        # found a model with better validation set accuracy

        snapshot_prefix = os.path.join(output_dir, prefix)
        snapshot_path = snapshot_prefix + 'model.pt'

        model.eval()
        val_iter.init_epoch()
        # calculate accuracy on validation set
        n_val_correct, val_loss = 0, 0
        actuals = []
        y_pred = []
        with torch.no_grad():
            for val_batch_idx, val_y in val_iter:
                pred_batch_y = model(val_batch_idx)

                pred_flat = torch.max(pred_batch_y, 1)[1].view(val_y.size())
                n_val_correct += (pred_flat == val_y).sum().item()
                actuals.extend(val_y)
                y_pred.extend(pred_flat)

        scores = self.score_evaluator(y_pred=y_pred, y_actual=actuals, pos_label=pos_label)

        current_score = scores[metric]
        if current_score > best_accuracy:
            self.logger.info("Snappshotting model becayse the current {} {} is greater than previous best {}".format(
                metric, current_score, best_accuracy))
            # save model, delete previous 'best_snapshot' files
            torch.save(model, snapshot_path)

            best_accuracy = current_score

        return best_accuracy
