import argparse
import itertools
import logging
import os
import sys

import numpy as np
import pandas as pd

from algorithms.RelationExtractionAverageFactory import RelationExtractionAverageFactory
from algorithms.RelationExtractionLinearDropoutWordFactory import RelationExtractorLinearNetworkDropoutWordFactory
from algorithms.RelationExtractionLinearFactory import RelationExtractionLinearFactory
from algorithms.RelationExtractorCnnNetwork import RelationExtractorCnnNetwork

networks_dict = {
    "Linear": RelationExtractionLinearFactory,
    "Avg": RelationExtractionAverageFactory,
    "LinearWithDropout": RelationExtractorLinearNetworkDropoutWordFactory,
    "Cnn": RelationExtractionLinearFactory,
}

model_dict = {
    "Cnn": RelationExtractorCnnNetwork
}


def prepare_data(interaction_type, file):
    data_df = pd.read_json(file)
    if interaction_type is not None:
        data_df = data_df.query('interactionType == "{}"'.format(interaction_type))
    labels = data_df[["isNegative"]]
    data_df = data_df[["pubmedabstract", "interactionType", "participant1Alias", "participant2Alias"]]
    data_df['participant1Alias'] = data_df['participant1Alias'].map(
        lambda x: ", ".join(list(itertools.chain.from_iterable(x))))
    data_df['participant2Alias'] = data_df['participant2Alias'].map(
        lambda x: ", ".join(list(itertools.chain.from_iterable(x))))
    labels = np.reshape(labels.values.tolist(), (-1,))
    return data_df, labels


def run(network, data_file, artifactsdir, out_dir, self_relations_filter=True):
    logger = logging.getLogger(__name__)

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

    logger.info("Running with self relations filter {}, network {}".format(self_relations_filter, network))

    logger.info("Loading from file {}".format(data_file))
    df = pd.read_json(data_file)
    logger.info("Data size after load: {}".format(df.shape))

    df_prep, labels = prepare_data(self_relations_filter, df)
    logger.info("Data size after prep: {}".format(df_prep.shape))

    network_factory = networks_dict[network]
    if network in model_dict:
        network_factory.model_network = model_dict[network]

    predictor = network_factory.load(artifactsdir)
    results = predictor(df_prep)

    df_prep["isValid"] = results

    predictions_file = os.path.join(out_dir, "predicted.json")
    select_columns = list(set(df.columns.values) - {"isValid"})

    final_df = df[select_columns].merge(df_prep[["isValid"]], how='inner', left_index=True, right_index=True)

    final_df.to_json(predictions_file)

    logger.info("Completed {}, {}".format(final_df.shape, final_df.columns.values))
    return results


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("network",
                        help="The type of network to use", choices=set(list(networks_dict.keys())))
    parser.add_argument("datajson",
                        help="The json data to predict")

    parser.add_argument("artefactsdir", help="The artefacts dir that contains model, vocab etc")
    parser.add_argument("outdir", help="The output dir")
    parser.add_argument("--self-filter", help="Filter self relations, if true remove self relations", type=bool,
                        default=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    results = run(args.network, args.datajson, args.artefactsdir,
                  args.outdir, args.self_filter)
