import argparse
import logging
import os
import sys

from torch.utils.data import DataLoader

from algorithms.CnnPosTrainInferenceBuilder import CnnPosTrainInferenceBuilder
from algorithms.ppiDataset import PPIDataset


def run(train_file, val_file, embedding_file, embed_dim, out_dir, epochs, interaction_type=None):
    logger = logging.getLogger(__name__)
    train = PPIDataset(train_file, interaction_type=interaction_type)
    val = PPIDataset(val_file, interaction_type=interaction_type)

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

    train_loader = DataLoader(train, shuffle=True)
    val_loader = DataLoader(val, shuffle=False)

    with open(embedding_file, "r") as embedding:
        # Ignore the first line as it contains the number of words and vector dim
        head = embedding.readline()
        logger.info("The embedding header is {}".format(head))
        builder = CnnPosTrainInferenceBuilder(dataset=train, embedding_dim=embed_dim, embedding_handle=embedding,
                                              output_dir=out_dir, epochs=epochs)
        train_pipeline = builder.get_trainpipeline()
        train_pipeline(train_loader, val_loader)


if "__main__" == __name__:
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

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    run(args.trainjson, args.valjson, args.embedding, args.embeddim,
        args.outdir, args.epochs, args.interaction_type)
