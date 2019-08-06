import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from algorithms.RelationExtractionAverageFactory import RelationExtractionAverageFactory
from algorithms.RelationExtractionLinearDropoutWordFactory import RelationExtractorLinearNetworkDropoutWordFactory
from algorithms.RelationExtractorCnnNetwork import RelationExtractorCnnNetwork
from algorithms.RelationExtractorCnnPosNetwork import RelationExtractorCnnPosNetwork
from algorithms.TrainInferencePipeline import TrainInferencePipeline

networks_dict = {
    "Linear": TrainInferencePipeline,
    "Avg": RelationExtractionAverageFactory,
    "LinearWithDropout": RelationExtractorLinearNetworkDropoutWordFactory,
    "Cnn": TrainInferencePipeline,
    "CnnPos": TrainInferencePipeline
}

model_dict = {
    "Cnn": RelationExtractorCnnNetwork,
    "CnnPos": RelationExtractorCnnPosNetwork
}


def prepare_data(self_relations_filter, data_df):
    logger = logging.getLogger(__name__)

    if self_relations_filter:
        logger.info("Removing self relations")

        data_df = data_df.query('participant1 != participant2')
    labels = data_df[["isValid"]]
    data_df = data_df[["abstract", "participant1", "participant2"]]

    labels = np.reshape(labels.values.tolist(), (-1,))
    return data_df, labels


def up_sample_minority(train_df, self_relations_filter):
    logger = logging.getLogger(__name__)

    # True is the minority class
    if self_relations_filter:
        logger.info("Removing self relations")
        train_df = train_df.query('participant1 != participant2')

    train_dat_0s = train_df.query('isValid == False')
    train_dat_1s = train_df.query('isValid == True')

    rep_1 = [train_dat_1s for x in range((train_dat_0s.shape[0] // train_dat_1s.shape[0]) // 3)]

    keep_1s = pd.concat(rep_1, axis=0)

    train_dat = pd.concat([keep_1s, train_dat_0s], axis=0)
    return train_dat


def run(network, train_file, val_file, embedding_file, embed_dim, out_dir, epochs, self_relations_filter=True,
        upsample=True):
    logger = logging.getLogger(__name__)

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

    class_size = 2

    logger.info("Running with self relations filter {}, upsample {}, network {}".format(self_relations_filter, upsample,
                                                                                        network))

    train_df = pd.read_json(train_file)
    logger.info("Original train size: {}, class distribution\n {}".format(train_df.shape,
                                                                          train_df['isValid'].value_counts()))
    val_df = pd.read_json(val_file)
    logger.info("Original validation size: {}, class distribution\n {}".format(val_df.shape,
                                                                               val_df[
                                                                                   'isValid'].value_counts()))
    if upsample:
        train_df = up_sample_minority(train_df, self_relations_filter)
        logger.info("Train size: {}, class distribution after upsampling \n{}".format(train_df.shape,
                                                                                      train_df[
                                                                                          'isValid'].value_counts()))

        val_df = up_sample_minority(val_df, self_relations_filter)
        logger.info("Validation size: {}, class distribution after upsampling \n{}".format(val_df.shape,
                                                                                           val_df[
                                                                                               'isValid'].value_counts()))

    train_df, train_labels = prepare_data(self_relations_filter, train_df)
    logger.info("Train size: {}, class distribution after data prep \n{}".format(train_df.shape,
                                                                                 np.bincount(
                                                                                     train_labels)))

    val_df, val_labels = prepare_data(self_relations_filter, val_df)

    logger.info("Training shape {}, test shape {}".format(train_df.shape, val_df.shape))

    with open(embedding_file, "r") as embedding:
        # Ignore the first line as it contains the number of words and vector dim
        head = embedding.readline()
        logger.info("The embedding header is {}".format(head))
        network_factory = networks_dict[network]

        # TODO: Constructor issue..Not all support class weights
        train_factory = network_factory(embedding_handle=embedding, embedding_dim=embed_dim,
                                        class_size=class_size,
                                        output_dir=out_dir, ngram=1, epochs=epochs, pos_label=True,
                                        class_weights_dict={True: 3, False: 1})
        if network in model_dict:
            train_factory.model_network = model_dict[network]

        model = train_factory(train_df, train_labels, val_df, val_labels)

        return model


if "__main__" == __name__:
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
    parser.add_argument("--self-filter", help="Filter self relations, if true remove self relations Y or N",
                        type=lambda x: (str(x).lower() == 'y'),
                        default=True)
    parser.add_argument("--upsample", help="Fix class imbalance when true, Y or N ",
                        type=lambda x: (str(x).lower() == 'y'),
                        default=False)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    run(args.network, args.trainjson, args.valjson, args.embedding, args.embeddim,
        args.outdir, args.epochs, args.self_filter, args.upsample)
