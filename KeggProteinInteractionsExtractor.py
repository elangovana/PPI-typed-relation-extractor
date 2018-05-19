import logging
import re
import pandas as pd

from bioservices import KEGG, UniProt


class KeggProteinInteractionsExtractor:

    def __init__(self):

        self._logger = logging.getLogger(__name__)
        # NOTE: This is  workaround to capture the problem with the logging moduel within Kegg which seems to reset all the log config
        log_level = logging.getLogger().getEffectiveLevel()
        self._logger.info("The log level is {} ".format(log_level))
        self.kegg = KEGG()
        new_level = logging.getLogger().getEffectiveLevel()
        self._logger.setLevel(log_level)
        self._logger.info("The new log level is {} ".format(new_level))


    def extract_protein_interaction(self, kegg_pathway_id):
        self._logger.info("Extracting PPIs for kegg pathway id {} ".format(kegg_pathway_id))

        kgml_parser = self.kegg.parse_kgml_pathway(kegg_pathway_id)
        protein_relations = list(filter(lambda d: d['link'] in ['PPrel'], kgml_parser['relations']))
        kegg_entries = kgml_parser['entries']
        result = []
        self._logger.debug("Extracting PPIs for kegg pathway id {} ".format(kegg_pathway_id))

        for rel in protein_relations:
            self._logger.debug("Extracting PPIs for kegg pathway id {} ".format(kegg_pathway_id))

            uniprot_dnumber = self._get_uniprot_numbers(self.kegg, kegg_entries, rel['entry2'])
            uniprot_snumber = self._get_uniprot_numbers(self.kegg, kegg_entries, rel['entry1'])

            for sv in uniprot_snumber:
                for dv in uniprot_dnumber:
                    result.append({"s_uniprot": sv, "s_gene_name": self._get_gene_names(sv), "interaction": rel['name'],
                                   "d_uniprot": dv, "d_genename": self._get_gene_names(dv)})

        # result in a dataframe
        result_df = pd.DataFrame(result)
        self._logger.info("Completed PPIs extraction for kegg pathway id {} ", kegg_pathway_id)
        return result_df

    def _get_uniprot_numbers(self, kegg, kegg_entries, entry_id):
        self._logger.info("Converting kegg Hsa numbers to uniprot ")
        regex_hsa = r"(?:\t)(.+)"
        uniprot_number = {}
        ko_number = list(filter(lambda d: d['id'] in [entry_id], kegg_entries))[0]['name']
        ko_number_map = kegg.link('hsa', ko_number)
        hsa_number_list = re.findall(regex_hsa, str(ko_number_map))
        if len(hsa_number_list) > 0:
            hsa_number = "+".join(hsa_number_list)
            uniprot_number = kegg.conv("uniprot", hsa_number)

        return map(lambda x: str(re.findall(r"(?:up:)(.+)", x)[0]), uniprot_number.values())

    def _get_gene_names(self, uniprot_list):
        self._logger.debug("Retrieving uniprot gene names for {}".format(' '.join(uniprot_list)))
        u = UniProt(verbose=False)
        gene_names_dict = u.mapping(fr="ACC,ID", to="GENENAME", query=uniprot_list)
        self._logger.debug("Retrieved uniprot gene names for {}".format(gene_names_dict))
        return ",".join(map(lambda x: ",".join(x), gene_names_dict.values()))
