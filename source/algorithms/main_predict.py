import argparse
import logging
import sys

from algorithms.InferencePipeline import InferencePipeline
from algorithms.dataset_factory import DatasetFactory


def run(dataset, datajson, artefactsbase_dir, outdir, positives_filter_threshold):
    dataset_factory = DatasetFactory().get_datasetfactory(dataset)
    dataset = dataset_factory.get_dataset(datajson)

    # artifacts_list = [os.path.join(artefactsbase_dir, d) for d in os.listdir(artefactsbase_dir) if os.path.isdir(d)]

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
