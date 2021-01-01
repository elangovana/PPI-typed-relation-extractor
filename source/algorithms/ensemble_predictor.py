import logging
from multiprocessing.dummy import Pool
from operator import add

import torch

from algorithms.Predictor import Predictor


class EnsemblePredictor:

    def __init__(self, model_wrapper=None):
        self.model_wrapper = model_wrapper or Predictor()

    def predict(self, model_networks, dataloader, device=None):
        if not self._is_iterable(model_networks):
            model_networks = [model_networks]

        if device is None:
            if torch.cuda.device_count() > 0:
                devices = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]
            else:
                devices = ["cpu"]
        else:
            devices = [device]

        # Use all available GPUS..
        self._logger.info("Using devices {}".format(devices))
        model_device_map = [(m, dataloader, devices[i % len(devices)]) for i, m in enumerate(model_networks)]
        with Pool(len(devices)) as p:
            agg_pred_scores = p.starmap(self.model_wrapper.predict, model_device_map)

        scores_ensemble = []
        for _, s in agg_pred_scores:
            self._populate_aggregate_scores_(s, scores_ensemble)

        scores_ensemble_avg = []
        # average the confidence
        for batch in scores_ensemble:
            scores_ensemble_avg.append([list(map(lambda x: x / len(model_networks), p)) for p in batch])

        # Predicted ensemble , arg max
        predicted_ensemble = []
        for batch in scores_ensemble_avg:
            predicted_ensemble.append([p.index(max(p)) for p in batch])
        return predicted_ensemble, scores_ensemble_avg

    @staticmethod
    def _populate_aggregate_scores_(batches_of_scores, output_scores_ensemble):

        for bi in range(len(batches_of_scores)):
            # First time not intialised case
            if len(output_scores_ensemble) < bi + 1:
                output_scores_ensemble.append(batches_of_scores[bi])
            else:
                batch_sum = []
                for line in range(len(batches_of_scores[bi])):
                    batch_sum.append(list(map(add, output_scores_ensemble[bi][line], batches_of_scores[bi][line])))
                output_scores_ensemble[bi] = batch_sum

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    @staticmethod
    def _is_iterable(o):
        try:
            iter(o)
        except TypeError:
            return False
        else:
            return True
