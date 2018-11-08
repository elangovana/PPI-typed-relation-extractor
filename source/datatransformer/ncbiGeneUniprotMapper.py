import logging
import urllib.request
from functools import lru_cache
from urllib.parse import urlencode


class NcbiGeneUniprotMapper:

    def __init__(self, url='https://www.uniprot.org/uploadlists/'):
        self.url = url
        self._cache_dict = {}

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def convert(self, ids):
        if type(ids) is list:
            return self._convert_list(ids)
        else:
            return self._convert(ids)

    def _convert_list(self, ids):
        ids = " ".join(ids)
        results = self._convert(ids)
        return results

    @lru_cache(125)
    def _convert(self, ids):
        params = {
            'from': 'P_ENTREZGENEID',
            'to': 'ACC',
            'format': 'tab',
            'query': ids
        }
        data = urlencode(params)
        response = urllib.request.urlopen("{}?{}".format(self.url, data))
        page = response.read().decode("utf-8")
        results = {}
        for r in page.split("\n")[1:]:
            if r.strip("\s") == "": break

            m = r.split("\t")

            results[m[0]] = m[2]
        return results
