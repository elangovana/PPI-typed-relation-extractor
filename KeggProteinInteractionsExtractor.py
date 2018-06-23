import logging
import re
from functools import lru_cache
import json
import pandas as pd

from bioservices import KEGG, UniProt


class KeggProteinInteractionsExtractor:

    def __init__(self):

        self._logger = logging.getLogger(__name__)
        self.kegg = KEGG()
        self.u = UniProt(verbose=False)
        self._cache_kegg_entry_uniprots = {}

    def extract_protein_interaction(self, kegg_pathway_id):
        self._logger.info("Extracting PPIs for kegg pathway id {} ".format(kegg_pathway_id))

        kgml = self.kegg.get(kegg_pathway_id, "kgml")
        result_df = self.extract_protein_interactions_kgml(kgml)

        # result in a dataframe
        self._logger.info("Completed PPIs extraction for kegg pathway id {} ".format(kegg_pathway_id))

        return result_df

    def extract_protein_interactions_kgml(self, kgml_string):
        self._logger.info("Parsing kgml")
        self._logger.debug(kgml_string)
        kgml_parser = self.kegg.parse_kgml_pathway(pathwayId='', res=kgml_string)
        result = []

        # Get PPRel (Protein protein relations) type from relation entries
        protein_relations = list(filter(lambda d: d['link'] in ['PPrel'], kgml_parser['relations']))

        for rel in protein_relations:
            self._logger.debug("Parsing relation for entry {}".format(rel))

            # Get the uniprot numbers corresponding to the 2 entries in the relation
            d_uniprot_numbers = self._cached_get_uniprot_numbers(rel['entry2'], kgml_parser)
            s_uniprot_numbers = self._cached_get_uniprot_numbers(rel['entry1'], kgml_parser)

            # Each source entry may map to multiple uniprot numbers, so loop through and get the relationships
            for s_uniprot in s_uniprot_numbers:
                # Same applies for the target entry in the relationship
                for d_uniprot in d_uniprot_numbers:
                    s_gene_name = self._get_gene_names(s_uniprot)
                    d_gene_name = self._get_gene_names(d_uniprot)
                    #set up key as the combination of the 2 interacting protein uniprot names in order
                    key ="#".join(sorted([s_uniprot, d_uniprot]))

                    # Add to result
                    rel_dict = {"key": key,"s_uniprot": s_uniprot, "s_gene_name": s_gene_name, "interaction": rel['name'],
                                "d_uniprot": d_uniprot, "d_genename": d_gene_name}
                    self._logger.debug("** Relation extracted {}".format(json.dumps(rel_dict)))
                    result.append(rel_dict)
        result_df = pd.DataFrame(result)
        return result_df

    def _cached_get_uniprot_numbers(self, entry_id, kgml_parser):
        if entry_id not in self._cache_kegg_entry_uniprots:
            # Uniprot numbers associated with the kegg entryid not in cache..
            # Note : The entry id is only unique within  a KGML file!!
            self._cache_kegg_entry_uniprots[entry_id] = self._get_uniprot_numbers(entry_id, kgml_parser)
        return self._cache_kegg_entry_uniprots[entry_id]

    def _get_uniprot_numbers(self, entry_id, kgml_parser):
        self._logger.debug("Converting kegg Hsa numbers to uniprot for entry id {}".format(entry_id))
        kegg_entries = kgml_parser['entries']
        hsa_uniprot_numbers_map = {}

        # Get the entry corresponding to the entry id
        # E.g entry id="49" name="ko:K00922 ko:K02649" type="ortholog"  ...
        matching_entries = list(filter(lambda d: d['id'] == entry_id, kegg_entries))
        if len(matching_entries) != 1:
            raise Exception("The number of entries for entry id {} should be 1, but is {}".format(entry_id, len(
                matching_entries)))
        entry = matching_entries[0]

        # Multiple KO numbers are separated by space, but the link query recognises that and returns corresponding HSA numbers
        # E.g name="ko:K00922 ko:K02649"
        ko_numbers_sep_space = entry['name']
        hsa_number_list = self.get_hsa_numbers(ko_numbers_sep_space)

        # Check if there are any HSA numbers associated with the KO numbers
        if len(hsa_number_list) > 0:
            hsa_number = "+".join(hsa_number_list)
            # Convert HSA to UniProt
            hsa_uniprot_numbers_map = self.kegg.conv("uniprot", hsa_number)

        self._logger.debug("HSA to Uniprot number map {}".format(json.dumps(hsa_uniprot_numbers_map)))
        kegg_uniprot_numbers = list(hsa_uniprot_numbers_map.values())
        # Remove the up: prefix from the uniprot numbers, as they look like 'up:B0LPE5', 'up:P31751', 'up:Q9Y243'
        result = list(map(lambda x: str(re.findall(r"(?:up:)(.+)", x)[0]), kegg_uniprot_numbers))
        self._logger.debug("Uniprot numbers {}".format(result))

        return result

    def get_hsa_numbers(self, ko_numbers_sep_space):
        # Assume the  Numbers are all Ko number or all HSA.
        self._logger.debug("Obtained KO numbers \n{}".format(ko_numbers_sep_space))

        # Undefined..
        if ko_numbers_sep_space.strip().lower() == "undefined":
            return []

        # If HSA then return as is..
        if ko_numbers_sep_space.strip().lower().startswith('hsa:'):
            return ko_numbers_sep_space.split(' ')

        # Check if ko.. else raise exception
        if not ko_numbers_sep_space.strip().lower().startswith('ko:'):
            raise ValueError("Expecting Ko or hsa numbers only, but found {}".format(ko_numbers_sep_space))

        # Get the HSA numbers (Homosapien proteins only for the KO)
        ko_number_map_sep_tab_sep_nl = self.kegg.link('hsa', ko_numbers_sep_space)
        # Extract just the HSA numbers from the multiline string individual maps
        # E.g
        # ko:K00922	hsa:5293
        # ko:K00922	hsa:5291
        # ko:K02649	hsa:5295
        self._logger.debug("HSA numbers for the KO numbers \n{}".format(ko_number_map_sep_tab_sep_nl))
        regex_hsa = r"(?:\t)(.+)"
        hsa_number_list = re.findall(regex_hsa, str(ko_number_map_sep_tab_sep_nl))
        return hsa_number_list

    @lru_cache(maxsize=100)
    def _get_gene_names(self, uniprot_number):
        # Get the gene names associated with the uniprot number
        self._logger.debug("Retrieving gene names for uniprotid {}".format(uniprot_number))
        gene_names_dict = self.u.mapping(fr="ACC,ID", to="GENENAME", query=uniprot_number)

        self._logger.debug("Gene names map : {}".format(json.dumps(gene_names_dict)))
        return ",".join(map(lambda x: ",".join(x), gene_names_dict.values()))
