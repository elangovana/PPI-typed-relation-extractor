import argparse
import logging
import os
import sys

from algorithms.TrainInferenceBuilder import TrainInferenceBuilder
from algorithms.dataset_mapper import str_to_dataset_class, get_datasets


def run(dataset_type, train_file, val_file, embedding_file, embed_dim, out_dir, epochs):
    logger = logging.getLogger(__name__)

    dataset_class = str_to_dataset_class(dataset_type)
    train, val = dataset_class(train_file), dataset_class(val_file)

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

    with open(embedding_file, "r") as embedding:
        # Ignore the first line as it contains the number of words and vector dim
        head = embedding.readline()
        logger.info("The embedding header is {}".format(head))
        builder = TrainInferenceBuilder(dataset=train, embedding_dim=embed_dim, embedding_handle=embedding,
                                        output_dir=out_dir, epochs=epochs)
        train_pipeline = builder.get_trainpipeline()
        train_pipeline(train, val)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="The dataset type", choices=get_datasets().keys())
    parser.add_argument("trainjson",
                        help="The input train json data")
    parser.add_argument("valjson",
                        help="The input val json data")
    parser.add_argument("embedding", help="The embedding file")
    parser.add_argument("embeddim", help="the embed dim", type=int)

    parser.add_argument("outdir", help="The output dir")
    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)
    parser.add_argument("--interaction-type", help="The interction type", default=None)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    run(args.dataset, args.trainjson, args.valjson, args.embedding, args.embeddim,
        args.outdir, args.epochs)
