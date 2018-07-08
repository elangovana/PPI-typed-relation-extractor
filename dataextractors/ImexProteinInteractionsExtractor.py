import logging
import re
from functools import lru_cache
import json
from xml.etree import ElementTree

import pandas as pd

from bioservices import UniProt


class ImexProteinInteractionsExtractor:

    def __init__(self, xmlfile, interactionlist=["phosphorylation"]):

        self.interactionlist = interactionlist
        self.xmlfile = xmlfile
        self.namespaces = {'df': 'http://psi.hupo.org/mi/mif'}  #
        self._logger = logging.getLogger(__name__)

        self.u = UniProt(verbose=False)
        self._cache_kegg_entry_uniprots = {}

    def extract_protein_interaction(self):
        self._logger.info("Extracting PPIs for kegg pathway id {} ".format(self.xmlfile))

        with open(self.xmlfile, "r") as handle:

            for entry in self._iter_elements_by_name(handle, "df:entry", self.namespaces):

                ele_interaction_list = entry.findall("df:interactionList/df:interaction", self.namespaces)
                for ele_interaction in ele_interaction_list:
                    interaction_type = ele_interaction.find("df:interactionType/df:names/df:shortLabel",
                                                            self.namespaces).text
                    if interaction_type not in self.interactionlist:
                        continue
                    i = 0




                    is_negative = ele_interaction.find("df:negative", self.namespaces).text
                    for ele_participant in ele_interaction.findall("df:participantList/df:participant", self.namespaces):
                        interfactor_ref_id = ele_participant.find( "df:interactorRef", self.namespaces).text

                        uniprotid = self.get_interactor_details(entry, interfactor_ref_id)

                        print("{},{}, {}, {}".format(interaction_type, interfactor_ref_id, uniprotid, is_negative))
                        i = i+1
                    print("Total participants in this interaction : {}".format(i))

        # result in a dataframe
        self._logger.info("Completed PPIs extraction for kegg pathway".format())
        result_df = pd.DataFrame()
        return result_df

    def _iter_elements_by_name(self, handle, name, namespace):
        events = ElementTree.iterparse(handle, events=("start", "end"))
        _, root = next(events)  # Grab the root element.

        expanded_name = name
        # If name has the namespace, expand it
        if name.index(":") >= 0:
            local_name = name[name.index(":") + 1:]
            namespace_short_name = name[:name.index(":")]
            expanded_name = "{{{}}}{}".format(namespace[namespace_short_name], local_name)

        for event, elem in events:
            if event == "end" and elem.tag == expanded_name:
                yield elem
                elem.clear()

    def get_interactor_details(self, entry, interfactor_ref_id):
        interactor_xpath = "df:interactorList/df:interactor[@id='{}']".format(interfactor_ref_id)
        ele_interactor = entry.find(interactor_xpath, self.namespaces)
        ele_unitprot = ele_interactor.find("df:xref/df:secondaryRef[@db='{}']".format("uniprotkb"), self.namespaces)
        if ele_unitprot is not None:
            return  ele_unitprot.attrib['id']
        return None
