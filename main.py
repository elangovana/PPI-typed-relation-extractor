import argparse
import sys
import logging
import os
import tempfile
from logging.config import fileConfig

from ExtractTrainingData import ExtractTrainingData
from KeggProteinInteractionsExtractor import KeggProteinInteractionsExtractor
from MIPSProteinInteractionsExtractor import MipsProteinInteractionsExtractor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kegg_pathway_id", help="Enter the kegg pathway id here, e.g path:ko05215")
    parser.add_argument("mips_ppi_xml_file", help="Enter the MIPS PPI XML file")
    args = parser.parse_args()

    # default kegg pathway id for sample test run
    kegg_pathway_id = args.kegg_pathway_id
    mips_xml_file = args.mips_ppi_xml_file

    # Configure logging
    fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))
    logger = logging.getLogger(__name__)

    # checking inputs..
    if not os.path.isfile(mips_xml_file):
        raise FileNotFoundError(
            "The file {} is not found. Please download the MIPS PPI interaction file and provide a path to the "
            "unzipped verison of the file".format(
                mips_xml_file))

    # Run Mips
    logger.info("Running MIPS extractor")
    ppi_extractor = MipsProteinInteractionsExtractor(mips_xml_file)
    result_df_mips = ppi_extractor.extract_protein_interaction()


    # Run extractor
    logger.info("Running kegg extractor")
    ppi_extractor = KeggProteinInteractionsExtractor()
    result_df_kegg = ppi_extractor.extract_protein_interaction(kegg_pathway_id)

    # Combine the results from MIPS & Kegg
    training_data_extractor = ExtractTrainingData(df_KeggPPI=result_df_kegg, df_MipsPPI=result_df_mips)
    result_df = training_data_extractor.run()

    # save result to file
    with  open(file="tmpKGML.csv", mode="w") as tmpfile:
        result_df.to_csv(tmpfile)
        logger.info("Writing output to {}".format(tmpfile.name))
