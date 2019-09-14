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
    parser.add_argument("--dataset", help="The dataset type", choices=get_datasets().keys(), required=True)

    parser.add_argument("--trainfile",
                        help="The input train file wrt to train  dir", required=True)

    parser.add_argument("--traindir",
                        help="The input train  dir", default=os.environ.get("SM_CHANNEL_TRAIN", "."))

    parser.add_argument("--valfile",
                        help="The input val file wrt to val  dir", required=True)

    parser.add_argument("--valdir",
                        help="The input val dir", default=os.environ.get("SM_CHANNEL_VAL", "."))

    parser.add_argument("--embeddingfile", help="The embedding file wrt to the embedding dir", required=True )

    parser.add_argument("--embeddingdir", help="The embedding dir", default=os.environ.get("SM_CHANNEL_EMBEDDING", "."))

    parser.add_argument("--outdir", help="The output dir", default=os.environ.get("SM_MODEL_DIR", "."))

    parser.add_argument("--embeddim", help="the embed dim", type=int, required=True)

    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)
    parser.add_argument("--interaction-type", help="The interction type", default=None)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    trainjson= os.path.join( args.traindir , args.trainfile)
    valjson = os.path.join(args.valdir, args.valfile)
    embeddingfile = os.path.join(args.embeddingdir, args.embeddingfile)
    run(args.dataset, trainjson, valjson, embeddingfile, args.embeddim,
        args.outdir, args.epochs)
