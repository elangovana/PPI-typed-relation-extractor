import argparse
import logging
import sys

import pandas as pd

from dataformatters.pubtatorAbstractOnlyFormatter import PubtatorAbstractOnlyFormatter


def run(input_json_file, output_json_file):
    logger = logging.getLogger(__name__)

    logger.info("Running inputs {}, {}".format(input_json_file, output_json_file))

    data = pd.read_json(input_json_file)
    abstract = lambda x: x.pubmedabstract
    pubbmed = lambda x: x.pubmedId

    with open(output_json_file, "w") as output_file:
        PubtatorAbstractOnlyFormatter()(data.itertuples(), pubbmed, abstract, output_file)


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("inputjson",
                        help="The input json data file")
    parser.add_argument("outfile", help="The output file name")
    args = parser.parse_args()

    run(args.inputjson, args.outfile)
