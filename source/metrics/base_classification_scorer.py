class BaseClassificationScorer:
    """
    Base results_scorer
    """

    def __call__(self, y_actual, y_pred, pos_label):
        raise NotImplementedError
