import json

import xmltodict


class ElasticSearchLoader:
    def convert_to_json(self, xmlHandle):
        return json.dumps(xmltodict.parse(xmlHandle))