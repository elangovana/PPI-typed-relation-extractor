import logging
import re
import pandas as pd

from bioservices import KEGG, UniProt


class KeggProteinInteractionsExtractor:

    def __init__(self):

        self._logger = logging.getLogger(__name__)
        log_level = logging.getLogger().getEffectiveLevel()
        self.kegg = KEGG()
        self.u = UniProt(verbose=False)
        self._cache_kegg_entry_uniprots = {}
        self._cache_uniprot_genename = {}

    def extract_protein_interaction(self, kegg_pathway_id):
        self._logger.info("Extracting PPIs for kegg pathway id {} ".format(kegg_pathway_id))

        kgml = self.kegg.get(kegg_pathway_id,"kgml")
        result = self.extract_protein_interactions_kgml(kgml)

        # result in a dataframe
        result_df = pd.DataFrame(result)
        self._logger.info("Completed PPIs extraction for kegg pathway id {} ".format(kegg_pathway_id))
        return result_df

    def extract_protein_interactions_kgml(self, kgml_string):
        kgml_parser = self.kegg.parse_kgml_pathway(pathwayId='', res=kgml_string)
        protein_relations = list(filter(lambda d: d['link'] in ['PPrel'], kgml_parser['relations']))
        kegg_entries = kgml_parser['entries']
        result = []

        for rel in protein_relations:
            self._logger.debug("Parsing relation for entry {}".format(rel))

            uniprot_dnumbers = self._cached_get_uniprot_numbers(self.kegg, kegg_entries, rel['entry2'])
            uniprot_snumbers = self._cached_get_uniprot_numbers(self.kegg, kegg_entries, rel['entry1'])

            for sv in uniprot_snumbers:
                for dv in uniprot_dnumbers:
                    result.append({"s_uniprot": sv, "s_gene_name": self._cached_get_gene_names(sv), "interaction": rel['name'],
                                   "d_uniprot": dv, "d_genename": self._cached_get_gene_names(dv)})
        return result

    def  _cached_get_uniprot_numbers(self, kegg, kegg_entries, entry_id):
       if entry_id not in self._cache_kegg_entry_uniprots :
            # Uniprot numbers associated with the kegg entryid not in cache..
            # Note : The entry id is only uniquie within  a KGML file!!
            self._cache_kegg_entry_uniprots[entry_id] = self._get_uniprot_numbers(kegg, kegg_entries, entry_id)
       return self._cache_kegg_entry_uniprots[entry_id]



    def _get_uniprot_numbers(self, kegg, kegg_entries, entry_id):
        self._logger.debug("Converting kegg Hsa numbers to uniprot for entry id {}".format(entry_id))
        regex_hsa = r"(?:\t)(.+)"
        uniprot_number = {}
        ko_number = list(filter(lambda d: d['id'] in [entry_id], kegg_entries))[0]['name']
        ko_number_map = kegg.link('hsa', ko_number)
        hsa_number_list = re.findall(regex_hsa, str(ko_number_map))
        if len(hsa_number_list) > 0:
            hsa_number = "+".join(hsa_number_list)
            uniprot_number = kegg.conv("uniprot", hsa_number)

        return map(lambda x: str(re.findall(r"(?:up:)(.+)", x)[0]), uniprot_number.values())

    def  _cached_get_gene_names(self, uniprot_number):
       if uniprot_number not in self._cache_uniprot_genename :
            # Uniprot numbers associated with the kegg entryid not in cache..
            # Note : The entry id is only uniquie within  a KGML file!!
            self._cache_uniprot_genename[uniprot_number] = self._get_gene_names(uniprot_number)
       return self._cache_uniprot_genename[uniprot_number]


    def _get_gene_names(self, uniprot_number):
        self._logger.debug("Retrieving uniprot gene names for {}".format(uniprot_number))
        gene_names_dict = self.u.mapping(fr="ACC,ID", to="GENENAME", query=uniprot_number)
        self._logger.debug("Retrieved uniprot gene names for {}".format(gene_names_dict))
        return ",".join(map(lambda x: ",".join(x), gene_names_dict.values()))
