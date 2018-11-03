import argparse
import logging
import sys

import numpy as np
import pandas as pd

from algorithms.RelationExtractionFactory import RelationExtractionFactory


def prepare_data(interaction_type, file):
    data_df = pd.read_json(file)
    if interaction_type is not None:
        data_df = data_df.query('interactionType == "{}"'.format(interaction_type))
    labels = data_df[["isNegative"]]
    data_df = data_df[["pubmedabstract", "interactionType", "destAlias", "sourceAlias"]]
    data_df['destAlias'] = data_df['destAlias'].map(lambda x: ", ".join(x[0]))
    data_df['sourceAlias'] = data_df['sourceAlias'].map(lambda x: ", ".join(x[0]))
    labels = np.reshape(labels.values.tolist(), (-1,))
    return data_df, labels


def run(train_file, val_file, embedding_file, embed_dim, tmp_dir, epochs, interaction_type=None):
    logger = logging.getLogger(__name__)

    class_size = 2

    logger.info("Running with interaction type {}".format(interaction_type))

    train_df, train_labels = prepare_data(interaction_type, train_file)
    val_df, val_labels = prepare_data(interaction_type, val_file)

    logger.info("Training shape {}, test shape {}".format(train_df.shape, val_df.shape))

    with open(embedding_file, "r") as embedding:
        # Ignore the first line as it contains the number of words and vector dim
        head = embedding.readline()
        logger.info("The embedding header is {}".format(head))
        train_factory = RelationExtractionFactory(embedding_handle=embedding, embedding_dim=embed_dim,
                                                  class_size=class_size,
                                                  output_dir=tmp_dir, ngram=1, epochs=epochs)
        train_factory(train_df, train_labels, val_df, val_labels)


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("trainjson",
                        help="The input train json data")
    parser.add_argument("valjson",
                        help="The input val json data")
    parser.add_argument("embedding", help="The embedding file")
    parser.add_argument("embeddim", help="the embed dim", type=int)

    parser.add_argument("outdir", help="The output dir")
    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)
    parser.add_argument("--interaction-type", help="The interction type", default=None)

    args = parser.parse_args()

    run(args.trainjson, args.valjson, args.embedding, args.embeddim,
        args.outdir, args.epochs, args.interaction_type)
