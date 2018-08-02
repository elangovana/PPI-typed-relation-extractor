import argparse
import logging
import os

from dataloader.ImexJsonProcessorFileWriter import ImexJsonProcessorFileWriter
from dataloader.imexDataPreprocessor import ImexDataPreprocessor


def bulk_run(data_dir, processor):
    logger = logging.getLogger(__name__)
    # Get xml files in dir
    for imex_file_name in os.listdir(data_dir):
        if not imex_file_name.endswith(".xml"):
            continue

        # Assuming all xml files are valid imex files.
        full_path = os.path.join(data_dir, imex_file_name)
        logger.info("Processing file {}".format(full_path))
        data_processor = ImexDataPreprocessor()

        with open(full_path, "rb") as xmlhandle:
            i = 0
            for doc in data_processor.run_pipeline(xmlhandle):
                i = i + 1
                processor.process(imex_file_name, i, doc)


class Processors:
    def __init__(self, processor_list):
        self.processor_list = processor_list

    def process(self, imex_file_name, doc_index, doc):
        for processor in self.processor_list:
            processor.process(imex_file_name, doc_index, doc)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help="The input directory containing the imex files")
    parser.add_argument("out_dir", help="The output dir")

    args = parser.parse_args()
    processor = ImexJsonProcessorFileWriter(args.out_dir)

    ##Consolidate all processors
    processors = Processors([processor])

    # Run
    bulk_run(args.input_dir, processors)
