from operator import add

from algorithms.Predictor import Predictor


class EnsemblePredictor:

    def __init__(self, model_wrapper=None):
        self.model_wrapper = model_wrapper or Predictor()

    def predict(self, model_networks, dataloader, device=None):
        if not self._is_iterable(model_networks):
            model_networks = [model_networks]

        scores_ensemble = []
        for model_network in model_networks:
            predicted, scores = self.model_wrapper(model_network, dataloader, device)

            self._populate_aggregate_scores_(predicted, scores, scores_ensemble)

        # average the confidence
        scores_ensemble = [list(map(lambda x: x / len(model_networks), p)) for p in scores_ensemble]

        # Predicted ensemble , arg max
        predicted_ensemble = [p.index(max(p)) for p in scores_ensemble]
        return predicted_ensemble, scores_ensemble

    @staticmethod
    def _populate_aggregate_scores_(predicted, scores, output_scores_ensemble):

        for i in range(len(predicted)):
            # First time not intialised case
            if len(output_scores_ensemble) < i + 1:
                output_scores_ensemble.append(scores[i])
            else:
                output_scores_ensemble[i] = list(map(add, output_scores_ensemble[i], scores[i]))

    @staticmethod
    def _is_iterable(o):
        try:
            iter(o)
        except TypeError:
            return False
        else:
            return True
