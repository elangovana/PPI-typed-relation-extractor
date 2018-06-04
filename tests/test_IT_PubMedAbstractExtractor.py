import unittest
import os

from ddt import ddt, data, unpack
from logging.config import fileConfig

from PubmedAbstractExtractor import PubmedAbstractExtractor


@ddt
class TestMipsProteinInteractionsExtractor(unittest.TestCase):

    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data(("sample_pubmed_data.xml", 2))
    @unpack
    def test_extract_protein_interactions_kgml(self, mips_file, expected_no_rel):
        #Arrange
        sut = PubmedAbstractExtractor()

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), mips_file), 'r') as myfile:
            #Act
            actual = sut.extract(myfile)

        #Assert
        self.assertEqual(expected_no_rel, len(actual))
