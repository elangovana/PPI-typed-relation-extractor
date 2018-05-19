import getopt
import sys

import logging

import os

from KeggProteinInteractionsExtractor import KeggProteinInteractionsExtractor

if __name__ == "__main__":
    # default kegg pathway id for sample test run
    kegg_pathway_id = "path:ko05215"
    try:
        opts, args = getopt.getopt(sys.argv, "hp", ["pathwayid="])
    except getopt.GetoptError:
        print 'main.py -p <kegg_pathway_id>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print '-p <kegg_pathway_id>'
            print 'Eg:'
            print '-p path:ko05215'
            sys.exit()
        elif opt in ("-p", "--pathwayid"):
            kegg_pathway_id = int(arg)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s.%(funcName)s: %(message)s')
    logger = logging.getLogger(__name__)

    # Run extractor
    logger.info("Running kegg extractor")
    ppi_extractor = KeggProteinInteractionsExtractor()
    result_df = ppi_extractor.extract_protein_interaction(kegg_pathway_id)

    # save result to file
    tmpfile = os.tmpfile()
    result_df.to_csv(tmpfile)
    logger.info("Writing output to {}".format(tmpfile))
