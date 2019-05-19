import argparse
import logging
import math
import os
import sys

import pandas as pd

from algorithms.RelationExtractionAverageFactory import RelationExtractionAverageFactory
from algorithms.RelationExtractionLinearDropoutWordFactory import RelationExtractorLinearNetworkDropoutWordFactory
from algorithms.RelationExtractionLinearFactory import RelationExtractionLinearFactory
from algorithms.RelationExtractorCnnNetwork import RelationExtractorCnnNetwork
from algorithms.RelationExtractorCnnPosNetwork import RelationExtractorCnnPosNetwork

networks_dict = {
    "Linear": RelationExtractionLinearFactory,
    "Avg": RelationExtractionAverageFactory,
    "LinearWithDropout": RelationExtractorLinearNetworkDropoutWordFactory,
    "Cnn": RelationExtractionLinearFactory,
    "CnnPos": RelationExtractionLinearFactory
}

model_dict = {
    "Cnn": RelationExtractorCnnNetwork,
    "CnnPos": RelationExtractorCnnPosNetwork
}


def prepare_data(data_df):
    data_df = data_df[["normalised_abstract", "interactionType", "participant1Id", "participant2Id"]]
    # data_df['participant1Alias'] = data_df['participant1Alias'].map(
    #     lambda x: ", ".join(list(itertools.chain.from_iterable(x))))
    # data_df['participant2Alias'] = data_df['participant2Alias'].map(
    #     lambda x: ", ".join(list(itertools.chain.from_iterable(x))))

    return data_df.copy(deep=True)


def run(network, data_file, artifactsdir, out_dir, postives_filter_threshold=0.0):
    logger = logging.getLogger(__name__)

    final_df = run_prediction(artifactsdir, data_file, network, out_dir)

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


def run_prediction(artifactsdir, data_file, network, out_dir):
    logger = logging.getLogger(__name__)

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

    logger.info("Running with self relations filter network {}".format(network))
    logger.info("Loading from file {}".format(data_file))
    df = pd.read_json(data_file)

    logger.info("Data size after load: {}".format(df.shape))
    df_prep = prepare_data(df)
    logger.info("Data size after prep: {}".format(df_prep.shape))

    network_factory = networks_dict[network]
    if network in model_dict:
        network_factory.model_network = model_dict[network]
    predictor = network_factory.load(artifactsdir)

    # Run prediction
    results, confidence_scores = predictor(df_prep)
    df_prep["predicted"] = results
    df_prep["confidence_scores"] = confidence_scores
    select_columns = df.columns.values
    final_df = df[select_columns].merge(df_prep[["predicted", "confidence_scores"]], how='inner', left_index=True,
                                        right_index=True)

    # This is log softmax, convert to softmax prob

    final_df["confidence_true"] = final_df.apply(lambda x: math.exp(x["confidence_scores"][True]), axis=1)
    final_df["confidence_false"] = final_df.apply(lambda x: math.exp(x["confidence_scores"][False]), axis=1)

    return final_df


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("network",
                        help="The type of network to use", choices=set(list(networks_dict.keys())))
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

    results = run(args.network, args.datajson, args.artefactsdir,
                  args.outdir, args.positives_filter_threshold)
