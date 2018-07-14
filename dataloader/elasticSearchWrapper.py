from __future__ import print_function

import csv

from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch import helpers


def connectES(esEndPoint, auth):
    print('Connecting to the ES Endpoint {0}'.format(esEndPoint))
    try:
        esClient = Elasticsearch(
            hosts=[{'host': esEndPoint, 'port': 443}],
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            http_auth=auth)
        return esClient
    except Exception as E:
        print("Unable to connect to {0}".format(esEndPoint))
        print(E)
        exit(3)


def createIndex(esClient, indexName, indexDoc = None):
    try:
        indexDoc = indexDoc or  {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
        index_exists = esClient.indices.exists(indexName)
        print("Index Exists ... {}".format(index_exists))
        if index_exists is False:
            esClient.indices.create(indexName, body=indexDoc)
            return 1
    except Exception as E:
        print("Unable to Create Index {0}".format(indexName))
        print(E)
        exit(4)


def indexBulkCsv(esClient, indexName, doc_type, filepath, fieldnames, delimiter='|', encoding='utf-8'):
    with open(filepath, encoding=encoding) as f:
        reader = csv.DictReader(f,  fieldnames=fieldnames, delimiter=delimiter)
        helpers.bulk(esClient, reader,  index=indexName, doc_type=doc_type, request_timeout=30)

def search(esClient, indexName, query, sort, size):
    res = esClient.search(index=indexName, body={"query": query, "size": size, "sort":sort})
    return res
           
