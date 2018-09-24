import argparse
import logging
import pathlib
import tempfile

import pandas as pd
import sys

from dataextractors.BulkImexProteinInteractionsExtractor import BulkImexProteinInteractionsExtractor
from datatransformer.ImexDataTransformerAugmentAbstract import ImexDataTransformerAugmentAbstract
from pipeline.OutputPathS3Mapper import OutputPathS3Mapper
from pipeline.pathLocalFileMapper import PathLocalFileMapper
from pipeline.inputPathS3Mapper import InputPathS3Mapper
from pipeline.simplePipeline import SimplePipeline


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



def run(input_dir, out_file):
    input_dir = path_rationalise(input_dir)
    local_out = get_localpath(out_file)

    pipeline = SimplePipeline()
    pipeline.pipeline_steps = [("Populate_Abstract", ImexDataTransformerAugmentAbstract())]

    data_extractor = BulkImexProteinInteractionsExtractor(["phosphorylation"])
    data_iter = data_extractor.get_protein_interactions(pathlib.Path(input_dir).glob('**/*.xml'))

    result = pipeline.run(data_iter)

    # write output as json
    pd.DataFrame(list(result)).to_json(local_out)

    upload_to_dest(local_out, out_file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help="The input directory containing the imex files")
    parser.add_argument("out_file", help="Output file")
    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    #Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    run(args.input_dir,
        args.out_file)
