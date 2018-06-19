import pandas as pd


class BulkKeggProteinInteractionsExtractor:
    def __init__(self, kegg_extractor=None):
        self.kegg_extractor = kegg_extractor

    def extract(self, pathway_list=None):
        if pathway_list is None:
            pathway_list = self.get_full_list()

        ppi_df_list = []
        for pathway in pathway_list:
            ppi_df = self.kegg_extractor.extract_protein_interaction(pathway)
            ppi_df_list.append(ppi_df)
        return pd.concat(ppi_df_list)

    def get_full_list(self):
        raise Exception("Not implemented")
        from bioservices import KEGG
        kegg = KEGG()
