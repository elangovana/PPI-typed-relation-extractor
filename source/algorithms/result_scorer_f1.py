"""

Calculate the score
"""


class ResultScorerF1:

    def __init__(self):
        pass

    def __call__(self, y_actual, y_pred, pos_label):
        from sklearn.metrics import f1_score

        f1 = f1_score(y_actual, y_pred, pos_label=pos_label, average='micro')

        return f1
