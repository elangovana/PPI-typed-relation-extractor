import logging
import re
from functools import lru_cache
import json
from xml.etree import ElementTree

import pandas as pd

from bioservices import  UniProt


class ImexProteinInteractionsExtractor:

    def __init__(self):

        self._logger = logging.getLogger(__name__)

        self.u = UniProt(verbose=False)
        self._cache_kegg_entry_uniprots = {}

    def extract_protein_interaction(self, kegg_pathway_id):
        self._logger.info("Extracting PPIs for kegg pathway id {} ".format(kegg_pathway_id))



        # result in a dataframe
        self._logger.info("Completed PPIs extraction for kegg pathway id {} ".format(kegg_pathway_id))

        return result_df

    def extract_protein_interactions_kgml(self, kgml_string):

        result_df = pd.DataFrame(result)
        self._logger.info("Extracted {} ppi relations".format(len(result_df)))
        return result_df

    def _cached_get_uniprot_numbers(self, entry_id, kgml_parser):

        return self._cache_kegg_entry_uniprots[entry_id]

    def _get_uniprot_numbers(self, entry_id, kgml_parser):


        return result

    def get_hsa_numbers(self, ko_numbers_sep_space):

        return hsa_number_list


    def _iter_elements_by_name(self, handle, name):
        events = ElementTree.iterparse(handle, events=("start", "end",))
        _, root = next(events)  # Grab the root element.
        for event, elem in events:
            if event == "end" and elem.tag == name:
                yield elem
                elem.clear()