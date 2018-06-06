import logging
import  pandas as pd

class ExtractTrainingData:
    def __init__(self, df_KeggPPI, df_MipsPPI):
        self.df_MipsPPI = df_MipsPPI
        self.df_KeggPPI = df_KeggPPI
        self._logger = logging.getLogger(__name__)

    def run(self):
        return pd.merge( self.df_KeggPPI, self.df_MipsPPI, on=['key','key'], how='inner')