import argparse
import glob
import logging
import os
import sys

from algorithms.InferencePipeline import InferencePipeline
from algorithms.dataset_factory import DatasetFactory


def run(dataset_name, datajson, artefactsbase_dir, outdir, positives_filter_threshold):
    logger = logging.getLogger(__name__)
    if os.path.isdir(datajson):
        for data_file in glob.glob("{}/*.json".format(datajson)):
            logger.info("Running prediction for {}".format(data_file))

            run_file(dataset_name, data_file, artefactsbase_dir, outdir, positives_filter_threshold)
    else:
        run_file(dataset_name, datajson, artefactsbase_dir, outdir, positives_filter_threshold)


def run_file(dataset_name, datajson, artefactsbase_dir, outdir, positives_filter_threshold):
    dataset_factory = DatasetFactory().get_datasetfactory(dataset_name)
    dataset = dataset_factory.get_dataset(datajson)
    InferencePipeline().run(dataset, datajson, artefactsbase_dir, outdir, positives_filter_threshold)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="The dataset type", choices=DatasetFactory().dataset_factory_names)

    parser.add_argument("datajson",
                        help="The json data to predict")

    parser.add_argument("artefactsdir", help="The base of artefacts dir that contains directories of model, vocab etc")
    parser.add_argument("outdir", help="The output dir")

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    parser.add_argument("--positives-filter-threshold", help="The threshold to filter positives", type=float,
                        default=0.0)

    args = parser.parse_args()

    print(args.__dict__)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    run(args.dataset, args.datajson, args.artefactsdir, args.outdir, args.positives_filter_threshold)
