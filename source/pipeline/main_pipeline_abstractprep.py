import argparse
import pathlib

import pandas as pd

from dataextractors.BulkImexProteinInteractionsExtractor import BulkImexProteinInteractionsExtractor
from datatransformer.ImexDataTransformerAugmentAbstract import ImexDataTransformerAugmentAbstract
from pipeline.simplePipeline import SimplePipeline


def run(input_dir, out_file):
    pipeline = SimplePipeline()
    pipeline.pipeline_steps = [("Populate_Abstract", ImexDataTransformerAugmentAbstract())]

    data_extractor = BulkImexProteinInteractionsExtractor(["phosphorylation"])
    data_iter = data_extractor.get_protein_interactions(pathlib.Path(input_dir).glob('**/*.xml'))

    result = pipeline.run(data_iter)

    #write output as json
    pd.DataFrame(list(result)).to_json(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help="The input directory containing the imex files")
    parser.add_argument("out_file", help="Output file")

    args = parser.parse_args()

    run(args.input_dir,
        args.out_file)
