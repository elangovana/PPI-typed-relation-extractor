import logging
import urllib.request
from functools import lru_cache
from urllib.parse import urlencode


class NcbiGeneUniprotMapper:

    def __init__(self, url='https://www.uniprot.org/uploadlists/'):
        self.url = url

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @lru_cache(maxsize=32)
    def convert(self, ids):
        if type(ids) is list:
            ids = " ".join(ids)

        params = {
            'from': 'P_ENTREZGENEID',
            'to': 'ACC',
            'format': 'tab',
            'query': ids
        }

        data = urlencode(params)
        response = urllib.request.urlopen("{}?{}".format(self.url, data))
        page = response.read().decode("utf-8")

        results = []
        for r in page.split("\n")[1:]:
            if r.strip("\s") == "": break

            m = r.split("\t")

            results.append({m[0]: m[2]})

        return results
