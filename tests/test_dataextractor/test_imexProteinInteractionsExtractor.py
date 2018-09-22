from logging.config import fileConfig
from unittest import TestCase

import os
from ddt import ddt, data, unpack

from dataextractors.ImexProteinInteractionsExtractor import ImexProteinInteractionsExtractor


@ddt
class TestImexProteinInteractionsExtractor(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data(("data/human_13_negative.xml", 0)
       ,("data/human_01.xml", 4))

    @unpack
    def test_extract_protein_interaction(self, xmlfile, expected_total):
        # Arrange
        full_xml_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), xmlfile)
        sut = ImexProteinInteractionsExtractor(['phosphorylation'])

        # Act
        actual =  list(sut.extract_protein_interaction(full_xml_file_path))

        # Assert
        self.assertEqual(len(actual), expected_total)
