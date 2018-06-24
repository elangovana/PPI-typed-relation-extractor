import unittest
import os

from ddt import ddt, data, unpack
from logging.config import fileConfig

from KeggProteinInteractionsExtractor import KeggProteinInteractionsExtractor


@ddt
class TestKeggProteinInteractionsExtractor(unittest.TestCase):

    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data(("sample_ko.kgml", 48)
          ,("sample_hsa.kgml", 3)
          ,("sample_nouniprot_forhsa.xml", 0))
    @unpack
    def test_extract_protein_interactions_kgml(self, kgml_file, expected_no_rel):
        #Arrange
        sut = KeggProteinInteractionsExtractor()
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), kgml_file), 'r') as myfile:
            kgml_string = myfile.read()

        #Act
        actual = sut.extract_protein_interactions_kgml(kgml_string)

        #Assert
        self.assertEqual(expected_no_rel, len(actual))
