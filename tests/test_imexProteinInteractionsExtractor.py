from logging.config import fileConfig
from unittest import TestCase

import os
from ddt import ddt, data, unpack

from dataextractors.ImexProteinInteractionsExtractor import ImexProteinInteractionsExtractor


@ddt
class TestImexProteinInteractionsExtractor(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data(("human_13_negative.xml", 500))
    @unpack
    def test_extract_protein_interaction(self, xmlfile, expected_total):
        #Arrange
        full_xml_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), xmlfile)
        sut = ImexProteinInteractionsExtractor(full_xml_file_path)

        #Assert
        sut.extract_protein_interaction()
