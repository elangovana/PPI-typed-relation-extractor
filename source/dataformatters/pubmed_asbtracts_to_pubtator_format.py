import argparse
import glob
import json
import logging
import os
import sys

from dataformatters.pubtatorAbstractOnlyFormatter import PubtatorAbstractOnlyFormatter

"""
Converts json formatted Pubmed abstracts to Pubtator format
"""


class PubmedAbstractsToPubtatorFormat:

    def __init__(self):
        self.pubtator_formatter = None

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def pubtator_formatter(self):
        self.__pubtator_formatter__ = self.__pubtator_formatter__ or PubtatorAbstractOnlyFormatter()
        return self.__pubtator_formatter__

    @pubtator_formatter.setter
    def pubtator_formatter(self, value):
        self.__pubtator_formatter__ = value

    def __call__(self, input: iter, output_handle):
        """
Converts the input  into pubtator format
        :type input: iter
        :param input: A iterable list of dict objects with keys article_abstract & pubmed_id
        :param output_handle:
        """

        abstract_extractor = lambda x: x["article_abstract"]
        pubbmed_extractor = lambda x: x["pubmed_id"]

        self.pubtator_formatter(input, pubbmed_extractor, abstract_extractor, output_handle)

    def from_dataframe(self, input_df, output_handle):
        """
Converts the input  is a data frame with columns article_abstract & pubmed_id
        :type input: DataFrame
        :param input: A dataframe containing columns article_abstract & pubmed_id
        :param output_handle:
        """

        abstract_extractor = lambda x: x.article_abstract
        pubbmed_extractor = lambda x: x.pubmed_id

        self.pubtator_formatter(input_df.itertuples(), pubbmed_extractor, abstract_extractor, output_handle)

    def read_json_file(self, input_file, output_handle):
        """
Reads an array of json from the input_file. The list of dict objects with keys article_abstract & pubmed_id
        :type input: str
        :param input: An input file containing an array of json
        :param output_handle:
        """

        with open(input_file, "r") as f:
            input = json.loads(f.read())
        self.__call__(input, output_handle)

    def read_json_files_dir(self, input_dir, output_dir):
        """
Reads an entire directory of json files
        :type input_dir: str
        :param input_dir: An input_dir containing json files
        :param output_dir:
        """
        files = glob.glob("{}/*.json".format(input_dir))
        for input_file in files:
            self.logger.info("Processing file {}".format(input_file))
            self._process_file(input_file, output_dir)

    def _process_file(self, input_file, output_dir):
        # Read input
        with open(input_file, "r") as f:
            input = json.loads(f.read())

        # Write out
        outfile = os.path.join(output_dir, "{}.txt".format(os.path.basename(input_file)))
        with open(outfile, "w") as output_handle:
            self.read_json_file(input, output_handle)


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir",
                        help="The input dir containing json files")
    parser.add_argument("outputdir", help="The output dir")
    args = parser.parse_args()

    PubmedAbstractsToPubtatorFormat().read_json_files_dir(args.inputdir, args.outputdir)
