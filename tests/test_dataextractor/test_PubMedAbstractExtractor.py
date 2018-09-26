import unittest
import os

from ddt import ddt, data, unpack
from logging.config import fileConfig

from dataextractors.PubmedAbstractExtractor import PubmedAbstractExtractor


@ddt
class TestPubmedAbstractExtractor(unittest.TestCase):

    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data(("data/sample_pubmed_data.xml", 2)
          ,("data/sample_pubmed_data_specialcharacter.xml",1))
    @unpack
    def test_extract(self, pubmed_resp, expected_no_rel):
        #Arrange
        sut = PubmedAbstractExtractor(pubmed_baseurl="http://localhost/invalid_mock_shoudl_not_be_is_use")


        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), pubmed_resp), 'r') as myfile:
            #Act
            actual = sut.extract(myfile)

        #Assert
        self.assertEqual(expected_no_rel, len(actual))
