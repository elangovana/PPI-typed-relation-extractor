import logging
import math
import os

import pandas as pd

from algorithms.TrainInferencePipeline import TrainInferencePipeline


class InferencePipeline:
    def run(self, dataset, data_file, artifactsdir, out_dir, postives_filter_threshold=0.0):
        logger = logging.getLogger(__name__)

        final_df = self.run_prediction(dataset, artifactsdir, data_file, out_dir)

        logger.info("Completed {}, {}".format(final_df.shape, final_df.columns.values))

        final_df = self._filter_threshold(final_df, postives_filter_threshold)

        predictions_file = os.path.join(out_dir, "predicted.json")
        final_df.to_json(predictions_file)

        return final_df

    def _filter_threshold(self, final_df, postives_filter_threshold):
        logger = logging.getLogger(__name__)

        if postives_filter_threshold == 0.0:
            return final_df

        logger.info(
            "Filtering True Positives with threshold > {}, currently {} records".format(postives_filter_threshold,
                                                                                        final_df.shape))
        final_df = final_df.query("confidence_true >= {}".format(postives_filter_threshold))
        logger.info("Post filter shape {}".format(final_df.shape))
        return final_df

    def run_prediction(self, dataset, artifactsdir, data_file, out_dir):
        logger = logging.getLogger(__name__)

        if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
            raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

        logger.info("Loading from file {}".format(data_file))

        df = pd.read_json(data_file)

        predictor = TrainInferencePipeline.load(artifactsdir)

        # Run prediction
        results, confidence_scores = predictor(dataset)
        df["predicted"] = results
        df["confidence_scores"] = confidence_scores

        # This is log softmax, convert to softmax prob

        df["confidence_true"] = df.apply(lambda x: math.exp(x["confidence_scores"][True]), axis=1)
        df["confidence_false"] = df.apply(lambda x: math.exp(x["confidence_scores"][False]), axis=1)

        return df

