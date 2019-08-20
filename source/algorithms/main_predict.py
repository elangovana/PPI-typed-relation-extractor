import argparse
import logging
import sys

from algorithms.InferencePipeline import InferencePipeline
from algorithms.dataset_mapper import get_datasets, str_to_dataset_class

if "__main__" == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="The dataset type", choices=get_datasets().keys())

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

    dataset_class = str_to_dataset_class(args.dataset)
    dataset = dataset_class(args.datajson)
    results = InferencePipeline().run(dataset, args.datajson, args.artefactsdir,
                                      args.outdir, args.positives_filter_threshold)
