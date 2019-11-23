from metrics.base_classification_scorer_factory import BaseClassificationScorerFactory
from metrics.result_scorer_f1_macro import ResultScorerF1Macro


class ResultScorerF1MacroFactory(BaseClassificationScorerFactory):
    """
    Factory for F1 results_scorer
    """

    def get(self):
        return ResultScorerF1Macro()
