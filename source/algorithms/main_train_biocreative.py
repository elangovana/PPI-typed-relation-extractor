import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from algorithms.RelationExtractionAverageFactory import RelationExtractionAverageFactory
from algorithms.RelationExtractionLinearFactory import RelationExtractionLinearFactory

networks_dict = {
    "Linear": RelationExtractionLinearFactory,
    "Avg": RelationExtractionAverageFactory
}


def prepare_data(self_relations_filter, file):
    data_df = pd.read_json(file)
    if self_relations_filter:
        data_df = data_df.query('participant1 == participant2')
    labels = data_df[["isValid"]]
    data_df = data_df[["abstract", "participant1", "participant2"]]

    labels = np.reshape(labels.values.tolist(), (-1,))
    return data_df, labels


def run(network, train_file, val_file, embedding_file, embed_dim, out_dir, epochs, self_relations_filter=True):
    logger = logging.getLogger(__name__)

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

    class_size = 2

    logger.info("Running with self relations filter {}, network {}".format(self_relations_filter, network))

    train_df, train_labels = prepare_data(self_relations_filter, train_file)
    val_df, val_labels = prepare_data(self_relations_filter, val_file)

    logger.info("Training shape {}, test shape {}".format(train_df.shape, val_df.shape))

    with open(embedding_file, "r") as embedding:
        # Ignore the first line as it contains the number of words and vector dim
        head = embedding.readline()
        logger.info("The embedding header is {}".format(head))
        network_factory = networks_dict[network]
        train_factory = network_factory(embedding_handle=embedding, embedding_dim=embed_dim,
                                        class_size=class_size,
                                        output_dir=out_dir, ngram=1, epochs=epochs)
        model = train_factory(train_df, train_labels, val_df, val_labels)

        return model


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("network",
                        help="The type of network to use", choices=set(list(networks_dict.keys())))
    parser.add_argument("trainjson",
                        help="The input train json data")
    parser.add_argument("valjson",
                        help="The input val json data")
    parser.add_argument("embedding", help="The embedding file")
    parser.add_argument("embeddim", help="the embed dim", type=int)

    parser.add_argument("outdir", help="The output dir")
    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)
    parser.add_argument("--self-filter", help="Filter self relations, if true remove self relations", type=bool,
                        default=True)

    args = parser.parse_args()

    run(args.network, args.trainjson, args.valjson, args.embedding, args.embeddim,
        args.outdir, args.epochs, args.self_filter)
