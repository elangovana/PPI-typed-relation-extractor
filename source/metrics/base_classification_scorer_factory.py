class BaseClassificationScorerFactory:
    """
    Base results_scorer factory
    """

    def get(self):
        raise NotImplementedError
