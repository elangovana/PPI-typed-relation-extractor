"""

Calculate the score
"""
from metrics.base_classification_scorer import BaseClassificationScorer
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

class ResultScorerAucMacro(BaseClassificationScorer):

    def __init__(self):
        pass

    def __call__(self, y_actual, y_pred, pos_label):
        y_pred = np.array(y_pred)
        y_actual =  np.array(y_actual)

        y_actual, y_pred = self.a_dd_fake_labels(y_actual, y_pred)

        f1 = roc_auc_score(y_actual, y_pred, average='macro', multi_class='ovr')

        return f1

    def _add_fake_labels(self, y_actual, y_pred):
        # HACK: Add fake records to get over not having enough true values in the validation / test set
        labels = list(range(y_pred.shape[-1]))
        unique_labels = np.unique(y_actual).tolist()
        missing_true = list(set(labels) - set(unique_labels))
        missing_actual_count = len(missing_true)
        true_label_size = y_pred.shape[-1]


        if missing_actual_count > 0:
            y_actual_extra = np.array(missing_true)

            y_actual = np.append(y_actual, y_actual_extra, axis=0)
            y_pred_extra = np.ones((missing_actual_count, true_label_size))
            norm_factor = np.sum(y_pred_extra, axis=-1)
            y_pred_extra = y_pred_extra / norm_factor[:, np.newaxis]
            y_pred = np.append(y_pred, y_pred_extra, axis=0)
        return y_actual, y_pred
