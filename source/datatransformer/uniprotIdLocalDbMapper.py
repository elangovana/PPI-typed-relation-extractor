import logging


class UniprotIdLocalDbMapper:
    """
    Converts a given Uniprot Id to target type names using a local mapping file. The file can be downloaded from ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/
    This file has three columns, delimited by tab:
    1. UniProtKB-AC
    2. ID_type
    3. ID
    """

    def __init__(self, handlelocaldb, type='UniProtKB-ID'):
        """
        :param handlelocaldb: Handle where each line is a 3 columns sep by tab containing UniProtKB-AC, ID_type and Id
        :param type: The ID_type to filter on. Only loads types match this
        """
        self.type = type
        self.localdb = handlelocaldb
        self._mapper = self._construct_mappers()

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def convert(self, ids):
        """
        A list of uniprot ids to convert to the target type
        :param ids:
        :return:
        """
        result = {}
        if type(ids) is str:
            result = self.populate_value(ids, result)
        else:
            for id in ids:
                result = self.populate_value(id, result)
        return result

    def populate_value(self, id, result):
        value = self._mapper.get(id, None)
        if value is not None: result[id] = value

        return result

    def _construct_mappers(self):
        result = {}

        for l in self.localdb:
            parts = l.split("\t")
            uniprot_id = parts[0].strip(" ")
            map_type = parts[1].strip(" ")
            target_id = parts[2].strip("\n").strip(" ")

            if map_type != self.type: continue

            matching_target_ids = result.get(uniprot_id, [])
            matching_target_ids.append(target_id)
            result[uniprot_id] = matching_target_ids

        return result
