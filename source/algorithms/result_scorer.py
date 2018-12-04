"""

Calculate the score
"""

score_type_recall = "recall"
score_type_precision = "precision"
score_type_f1 = "f-score"
score_type_accuracy = "accuracy"


class ResultScorer:

    def __call__(self, y_actual, y_pred, pos_label):
        from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

        recall, precision, f1 = recall_score(y_actual, y_pred, pos_label=pos_label), precision_score(y_actual, y_pred,
                                                                                                     pos_label=pos_label), f1_score(
            y_actual, y_pred, pos_label=pos_label)

        accuracy = accuracy_score(y_actual, y_pred)

        return {score_type_recall: recall, score_type_precision: precision, score_type_f1: f1,
                score_type_accuracy: accuracy}
