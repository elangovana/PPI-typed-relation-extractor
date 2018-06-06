import unittest
import os

from ddt import ddt, data, unpack
from logging.config import fileConfig

from MIPSProteinInteractionsExtractor import MipsProteinInteractionsExtractor


@ddt
class TestMipsProteinInteractionsExtractor(unittest.TestCase):

    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data(("sample_mips.xml", 4))
    @unpack
    def test_extract_protein_interactions_kgml(self, mips_file, expected_no_rel):
        #Arrange
        full_mips_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), mips_file)
        sut = MipsProteinInteractionsExtractor(full_mips_file_path)

        with open(full_mips_file_path, 'r') as myfile:
            #Act
            actual = sut.extract_protein_interaction_file(myfile)


        self.assertEqual(expected_no_rel, len(actual))
