import argparse
import logging
import pathlib
import tempfile

import pandas as pd
import sys

from dataextractors.BulkImexProteinInteractionsExtractor import BulkImexProteinInteractionsExtractor
from datatransformer.ImexDataTransformerAugmentAbstract import ImexDataTransformerAugmentAbstract
from pipeline.OutputPathS3Mapper import OutputPathS3Mapper
from pipeline.dataPrepPipeline import DataPrepPipeline
from pipeline.pathLocalFileMapper import PathLocalFileMapper
from pipeline.inputPathS3Mapper import InputPathS3Mapper


def path_rationalise(path):
    # check if path uri
    index = path.find("://")

    ## local file
    if index == -1: return path

    scheme = path[0:index]

    scheme_mapper = {
        "file": PathLocalFileMapper()
        , "s3": InputPathS3Mapper()}

    return scheme_mapper[scheme](path)


def upload_to_dest(localpath, remotpath):
    # check if path uri
    index = remotpath.find("://")

    ## local file
    if index == -1: return remotpath

    scheme = remotpath[0:index]

    scheme_mapper = {
        "file": lambda r, l: l
        , "s3": OutputPathS3Mapper()}

    return scheme_mapper[scheme](remotpath, localpath)


def get_localpath(path):
    # check if path uri
    index = path.find("://")

    ## local file
    if index == -1: return path

    # Extract just the path
    scheme = path[0:index]
    if scheme.lower() == "file":
        return path[index + 3:]

    # Some other protocol.. So create temp file path
    # TODO handle dir vs file path checks

    tempath = tempfile.mkstemp("data")[1]

    return tempath


def run(input_dir, out_file, interaction_types):
    logger = logging.getLogger(__name__)

    input_dir = path_rationalise(input_dir)
    local_out = get_localpath(out_file)

    pipeline = DataPrepPipeline()
    pipeline.pipeline_steps = [("Populate_Abstract", ImexDataTransformerAugmentAbstract())]

    logger.info("Extracting interaction types {}".format(",".join(interaction_types)))
    data_extractor = BulkImexProteinInteractionsExtractor(interaction_types)
    data_iter = data_extractor.get_protein_interactions(pathlib.Path(input_dir).glob('**/*.xml'))

    result = pipeline.run(data_iter)

    # write output as json
    pd.DataFrame(list(result)).to_json(local_out)

    upload_to_dest(local_out, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    interaction_types_csv = ",".join(['phosphorylation', 'dephosphorylation', 'ubiquitination',
                                      'methylation', 'acetylation', 'deubiquitination', 'demethylation'])
    parser.add_argument("input_dir",
                        help="The input directory containing the imex files")
    parser.add_argument("out_file", help="Output file")
    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    parser.add_argument("--interaction-types",
                        help="A comma separated list of interactions. The interaction type must match the imex types. For more details see https://www.ebi.ac.uk/intact/validator/help.xhtml and https://www.ebi.ac.uk/ols/ontologies/mi/terms?obo_id=MI%3A0190",
                        default=interaction_types_csv)

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.info("Starting run with arguments...\n{}".format(args.__dict__))

    ## Run
    interaction_types = [i.strip() for i in args.interaction_types.split(",") if i is not None and i.strip() != ""]
    run(args.input_dir, args.out_file, interaction_types)
    logger.info("Completed run...")
