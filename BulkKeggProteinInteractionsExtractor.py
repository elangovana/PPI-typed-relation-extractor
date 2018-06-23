import logging

import pandas as pd

from KeggProteinInteractionsExtractor import  KeggProteinInteractionsExtractor


class BulkKeggProteinInteractionsExtractor:
    def __init__(self, kegg_extractor=None):
        self.kegg_extractor = kegg_extractor or KeggProteinInteractionsExtractor()
        self._logger = logging.getLogger(__name__)

    def extract(self, pathway_list=None):
        if pathway_list is None:
            pathway_list = self.extract_all().keys()
        self._logger.info("Extracting for debug\n {}".format("\n".join(pathway_list)))

        ppi_df_list = []
        for pathway in pathway_list:
            ppi_df = self.kegg_extractor.extract_protein_interaction(pathway)
            ppi_df_list.append(ppi_df)

        concated_df = pd.concat(ppi_df_list)
        #remove duplicates
        concated_df.drop_duplicates(subset=['key'], keep="first")
        return concated_df

    def extract_all(self):
        from bioservices import KEGG
        kegg = KEGG()
        pathway_list = filter(None, kegg.list("pathway/hsa").split("\n"))

        pathway_dict ={}
        for p in pathway_list:
            id = p.split("\t")[0]
            name = p.split("\t")[1]

            pathway_dict[id]=name

        return pathway_dict
