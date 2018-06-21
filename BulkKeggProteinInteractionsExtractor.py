import pandas as pd


class BulkKeggProteinInteractionsExtractor:
    def __init__(self, kegg_extractor=None):
        self.kegg_extractor = kegg_extractor

    def extract(self, pathway_list=None):
        if pathway_list is None:
            pathway_list = self.extract_all().keys()

        ppi_df_list = []
        for pathway in pathway_list:
            ppi_df = self.kegg_extractor.extract_protein_interaction(pathway)
            ppi_df_list.append(ppi_df)
        return pd.concat(ppi_df_list)

    def extract_all(self):
        from bioservices import KEGG
        kegg = KEGG()
        pathway_list = filter(None, kegg.list("pathway").split("\n"))

        pathway_dict ={}
        for p in pathway_list:
            id = p.split("\t")[0]
            name = p.split("\t")[1]

            pathway_dict["id"]=id
            pathway_dict["name"]= name
        return pathway_dict
