# coding=utf-8
import logging

import os
import tempfile

import pandas as pd
import requests
import xml.etree.cElementTree as ElementTree

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

            # Start Extracting PPIs
            self.extract_protein_interaction_file(tmpfile)

    def extract_protein_interaction_file(self, handle):
        """
        Extracts PPI interactions from a MIPS file
        :param mips_file:
        """
        self._logger.info("Running extract_protein_interaction_file...")
        result_arr = []
        for interaction in self._iter_elements_by_name(handle, "interaction"):

            # Find Pub Med Rreferences
            #TODO
            ele_ref_list = interaction.findall("experimentList/experimentDescription/bibref/xref/primaryRef")
            #Get Protein Participants
            ele_participant_list = interaction.findall("participantList/proteinParticipant/proteinInteractor")

            #loop through the ref list
            for ele_ref in ele_ref_list:
                doc_id = ele_ref.attrib.get("id", "")
                doc_type = ele_ref.attrib.get("db","")
                for s_ele_participant in ele_participant_list:
                    s_protien_name = s_ele_participant.find("names/fullName").text
                    s_protien_id = s_ele_participant.find("xref/primaryRef").attrib.get( "id", "")
                    s_protien_db = s_ele_participant.find("xref/primaryRef").attrib.get("db", "")

                    for d_ele_participant in ele_participant_list:

                        d_protien_name = d_ele_participant.find("names/fullName").text
                        d_protien_id = d_ele_participant.find("xref/primaryRef").attrib.get("id", "")
                        d_protien_db = d_ele_participant.find("xref/primaryRef").attrib.get("db", "")

                        # Assume no self relations
                        if s_protien_id == d_protien_id and s_protien_id != "":
                            continue

                        # set up key as the combination of the 2 interacting protein uniprot names in order
                        key = "#".join(sorted([s_protien_id, d_protien_id]))

                        interaction = {}
                        interaction["key"] = key
                        interaction["doc_id"] = doc_id
                        interaction["doc_type"] = doc_type
                        interaction["s_protien_id"] = s_protien_id
                        interaction["s_protien_name"] = s_protien_name
                        interaction["s_protien_db"] = s_protien_db

                        interaction["d_protien_id"] = d_protien_id
                        interaction["d_protien_name"] = d_protien_name
                        interaction["d_protien_db"] = d_protien_db

                        result_arr.append(interaction)
                        self._logger.debug("{} {} {} {} {} {}".format(doc_id, doc_type, s_protien_id, s_protien_db, d_protien_id, d_protien_db))

        df_result = pd.DataFrame(result_arr)
        return df_result

    def _iter_elements_by_name(self, handle, name):
        events = ElementTree.iterparse(handle, events=("start", "end",))
        _, root = next(events)  # Grab the root element.
        for event, elem in events:
            if event == "end" and elem.tag == name:
                yield elem
                elem.clear()


