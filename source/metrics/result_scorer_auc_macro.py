"""

Calculate the score
"""
from metrics.base_classification_scorer import BaseClassificationScorer
from sklearn.metrics import roc_auc_score


class ResultScorerAucMacro(BaseClassificationScorer):

    def __init__(self):
        pass

    def __call__(self, y_actual, y_pred, pos_label):
        f1 = roc_auc_score(y_actual, y_pred, average='macro', multi_class='ovr')

        return f1
