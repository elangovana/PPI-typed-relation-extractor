from datetime import datetime
from elasticsearch import Elasticsearch

from dataloader.elasticSearchWrapper import connectES, createIndex


class ImexJsonProcessorElasticSearchLoader:

    def __init__(self, esclient=None, index="imexdocs_index"):
        self.esclient = esclient
        self.index = index
        self._index_exists = False

    @property
    def esclient(self):
        self.__esclient = self.__esclient or connectES()
        return self.__esclient

    @esclient.setter
    def esclient(self, client):
        self.__esclient = client

    def initialse(self):
        if not self._index_exists :

            createIndex(self.esclient, self.index)
            self._index_exists = True

    def process(self, imex_file_name, doc_index, doc):
        self.initialse()
        id = "{}_{:03d}.json".format(imex_file_name, doc_index).lower()
        res = self.esclient.index(index=self.index, doc_type='imex', id=id, body=doc)
        print(res['result'])
