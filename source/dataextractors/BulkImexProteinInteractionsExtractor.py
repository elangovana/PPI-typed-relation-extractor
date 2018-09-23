import logging
from dataextractors.ImexProteinInteractionsExtractor import ImexProteinInteractionsExtractor


class BulkImexProteinInteractionsExtractor:

    def __init__(self, interactionlist=None):
        self.interactionlist = interactionlist or ["phosphorylation"]

    @property
    def imexProteinInteractionsExtractor(self):
        self.__imexProteinInteractionsExtractor__ = self.__imexProteinInteractionsExtractor__ or ImexProteinInteractionsExtractor(self.interactionlist)
        return self.__imexProteinInteractionsExtractor__

    @imexProteinInteractionsExtractor.setter
    def imexProteinInteractionsExtractor(self, value):
        self.__imexProteinInteractionsExtractor__ = value

    @property
    def logger(self):
        return logging.getLogger(__name__)


    def get_protein_interactions(self, filelist_iter):
        """
Extracts protein interactions from a list of files
        :param filelist_iter:
        """
        self.logger.info("Extracting interactions from path")

        for imex_file_name in filelist_iter:
            if not imex_file_name.endswith(".xml"):
                continue

            self.logger.info("Processing file {}".format(imex_file_name))
            yield from self.imexProteinInteractionsExtractor.extract_protein_interaction(imex_file_name)




