# coding=utf-8
import logging

import os
import tempfile

import pandas as pd
import requests

"""
Extracts PPI from MIPS http://mips.helmholtz-muenchen.de/proj/ppi/, http://mips.helmholtz-muenchen.de/proj/ppi/data/mppi.gz
Cite
Pagel P, Kovac S, Oesterheld M, Brauner B, Dunger-Kaltenbach I, Frishman G, Montrone C, Mark P, Stümpflen V, Mewes HW, Ruepp A, Frishman D
The MIPS mammalian protein-protein interaction database
Bioinformatics 2005; 21(6):832-834; [Epub 2004 Nov 5]   doi:10.1093/bioinformatics/bti115  
"""


class MipsProteinInteractionsExtractor:
    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def extract_protein_interaction(self, uri="http://mips.helmholtz-muenchen.de/proj/ppi/data/mppi.gz"):
        self._logger.info("Extracting protein extractions")

        # Downloading PPI Xml file
        r = requests.get(uri, allow_redirects=True)
        with tempfile.TemporaryFile(suffix=".csv", mode="w+r") as tmpfile:
            self._logger.info("Downloading {} to temp file".format(uri))
            tmpfile.write(r.content)
            tmpfile.seek(0)





