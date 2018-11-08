import logging

"""
Converts to uniprot numbers using a local mapping file. The file can be downloaded from ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/
This file has three columns, delimited by tab:
1. UniProtKB-AC 
2. ID_type 
3. ID
"""


class NcbiGeneUniprotLocalDbMapper:

    def __init__(self, handlelocaldb, type='GeneID'):
        """
        :param handlelocaldb: Handle where each line is a 3 columns sep by tab containing UniProtKB-AC, ID_type and Id
        :param type: The ID_type to filter on. Only loads types match this
        """
        self.type = type
        self.localdb = handlelocaldb
        self.__mapper__ = None

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def mapper(self):
        self.__mapper__ = self.__mapper__ or self._construct_mapper()
        return self.__mapper__

    def convert(self, ids):
        result = {}
        if type(ids) is str:
            result = self.populate_value(ids, result)
        else:
            for id in ids:
                result = self.populate_value(id, result)
        return result

    def populate_value(self, id, result):
        value = self.mapper.get(id, None)
        if value is not None: result[id] = value

        return result

    def _construct_mapper(self):
        result = {}
        # ignore head
        next(self.localdb)
        next(self.localdb)
        for l in self.localdb:
            entries = l.split("\t")
            if entries[1] != self.type: continue
            result[entries[2].strip("\n")] = entries[0]

        return result
