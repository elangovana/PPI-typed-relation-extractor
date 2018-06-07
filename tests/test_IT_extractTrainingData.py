from unittest import TestCase
import unittest
import os

from ddt import ddt, data, unpack
from logging.config import fileConfig

from ExtractTrainingData import ExtractTrainingData
from KeggProteinInteractionsExtractor import KeggProteinInteractionsExtractor
from MIPSProteinInteractionsExtractor import MipsProteinInteractionsExtractor


@ddt
class TestExtractTrainingData(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data(("sample_ko.kgml", "sample_mips.xml", 0))
    @unpack
    def test_run(self, kgml_file, mips_file, expected_no_records):
        # Arrange
        #kegg
        kegg = KeggProteinInteractionsExtractor()
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), kgml_file), 'r') as myfile:
            kgml_string = myfile.read()
        df_kegg = kegg.extract_protein_interactions_kgml(kgml_string)

        # Mips
        mips_full_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), mips_file)
        mips = MipsProteinInteractionsExtractor(mips_full_file_path)
        with open(mips_full_file_path, 'r') as myfile:
            df_mips = mips.extract_protein_interaction_file(myfile)

        sut = ExtractTrainingData(df_kegg, df_mips)

        # Act
        actual = sut.run()

        #Assert
        self.assertEqual(expected_no_records, len(actual))

