import argparse
import logging
import os


from datavisualiser.ImexJsonProcessorFileWriter import ImexJsonProcessorFileWriter
from datavisualiser.Processors import Processors
from datavisualiser.imexDataPreprocessorFlattenXml import ImexDataPreprocessorFlattenXml


def bulk_run(data_dir, processor):
    logger = logging.getLogger(__name__)
    # Get xml files in dir
    for imex_file_name in os.listdir(data_dir):
        if not imex_file_name.endswith(".xml"):
            continue

        # Assuming all xml files are valid imex files.
        full_path = os.path.join(data_dir, imex_file_name)
        logger.info("Processing file {}".format(full_path))
        data_processor = ImexDataPreprocessorFlattenXml()

        with open(full_path, "rb") as xmlhandle:
            i = 0
            for doc in data_processor.run_pipeline(xmlhandle):
                i = i + 1
                processor.process(imex_file_name, i, doc)


def run(input_dir, out_dir):


    ##Consolidate all processors
    filewriter_processor = ImexJsonProcessorFileWriter(out_dir)
    processors = Processors([filewriter_processor])

    # Run
    bulk_run(input_dir, processors)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help="The input directory containing the imex files")
    parser.add_argument("out_dir", help="The output dir")

    args = parser.parse_args()

    run(args.input_dir,
        args.out_dir)
