import unittest
import os

from ddt import ddt, data, unpack
from logging.config import fileConfig

from dataextractors.KeggProteinInteractionsExtractor import KeggProteinInteractionsExtractor
from datatransformer.ImexJsonProcessorElasticSearchLoader import ImexJsonProcessorElasticSearchLoader
from aws_requests_auth.aws_auth import AWSRequestsAuth

from datatransformer.elasticSearchWrapper import connectES

'''
Prerequistes.. This is an integration test. Requires the Elastic search environment to be set up. 
Make sure the following environment variables are available

        # export elasticsearch_domain_name  = 'your-elastic-search-endpoint.us-east-2.es.amazonaws.com'
        # export AWS_REGION = 'us-east-2'
        # export AWS_ACCESS_KEY_ID = "***"
        # export AWS_SECRET_ACCESS_KEY ="******"
 

'''


@ddt
class TestITImexJsonProcessorElasticSearchLoader(unittest.TestCase):

    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

        # es
        esdomain = os.environ['elasticsearch_domain_name']
        region = os.environ['AWS_REGION']
        print(region)
        auth = AWSRequestsAuth(aws_access_key=os.environ['AWS_ACCESS_KEY_ID'],
                               aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                               aws_token="",
                               aws_host=esdomain,
                               aws_region=region,
                               aws_service='es')
        self._esClient = connectES(esdomain, auth)


    @data(("data/humain_01.xml_065.json")
          )
    @unpack
    def test_extract_protein_interactions_kgml(self, kgml_file):
        #Arrange
        sut = ImexJsonProcessorElasticSearchLoader(self._esClient)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), kgml_file), 'r') as myfile:
            json_string = myfile.read()

        #Act
        actual = sut.process("test", 65, json_string)

        #Assert

