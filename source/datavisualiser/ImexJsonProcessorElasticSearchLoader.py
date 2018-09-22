import logging
from datavisualiser.elasticSearchWrapper import createIndex


class ImexJsonProcessorElasticSearchLoader:

    def __init__(self, esclient, index="imexdocs_index"):
        self.esclient = esclient
        self.index = index
        self._index_exists = False
        self._logger = logging.getLogger(__name__)

    @property
    def esclient(self):
        self.__esclient = self.__esclient
        return self.__esclient

    @esclient.setter
    def esclient(self, client):
        self.__esclient = client

    def initialse(self):
        if not self._index_exists:
            createIndex(self.esclient, self.index)
            self._index_exists = True

    def process(self, imex_file_name, doc_index, doc):
        self.initialse()
        id = "{}_{:03d}.json".format(imex_file_name, doc_index).lower()
        self.esclient.index(index=self.index, doc_type='imex', id=id, body=doc)
