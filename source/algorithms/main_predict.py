import argparse
import logging
import math
import os
import sys

import pandas as pd

from algorithms.PpiDataset import PPIDataset
from algorithms.TrainInferencePipeline import TrainInferencePipeline


def run(data_file, artifactsdir, out_dir, postives_filter_threshold=0.0):
    logger = logging.getLogger(__name__)

    final_df = run_prediction(artifactsdir, data_file, out_dir)

    logger.info("Completed {}, {}".format(final_df.shape, final_df.columns.values))

    if postives_filter_threshold > 0.0:
        logger.info(
            "Filtering True Positives with threshold > {}, currently {} records".format(postives_filter_threshold,
                                                                                        final_df.shape))
        final_df = final_df.query("confidence_true >= {}".format(postives_filter_threshold))
        logger.info("Post filter shape {}".format(final_df.shape))

    predictions_file = os.path.join(out_dir, "predicted.json")
    final_df.to_json(predictions_file)

    return final_df


def run_prediction(artifactsdir, data_file, out_dir):
    logger = logging.getLogger(__name__)

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

    logger.info("Loading from file {}".format(data_file))

    df = pd.read_json(data_file)

    predictor = TrainInferencePipeline.load(artifactsdir)
    val_dataset = PPIDataset(data_file)

    # Run prediction
    results, confidence_scores = predictor(val_dataset)
    print(confidence_scores)
    df["predicted"] = results
    df["confidence_scores"] = confidence_scores

    # This is log softmax, convert to softmax prob

    df["confidence_true"] = df.apply(lambda x: math.exp(x["confidence_scores"][True]), axis=1)
    df["confidence_false"] = df.apply(lambda x: math.exp(x["confidence_scores"][False]), axis=1)

    return df


if "__main__" == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("datajson",
                        help="The json data to predict")

    parser.add_argument("artefactsdir", help="The artefacts dir that contains model, vocab etc")
    parser.add_argument("outdir", help="The output dir")

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    parser.add_argument("--positives-filter-threshold", help="The threshold to filter positives", type=float,
                        default=0.0)

    args = parser.parse_args()

    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    results = run(args.datajson, args.artefactsdir,
                  args.outdir, args.positives_filter_threshold)
