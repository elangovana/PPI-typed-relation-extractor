import os
import unittest
from logging.config import fileConfig

from ddt import ddt

from dataextractors.PubmedAbstractExtractor import PubmedAbstractExtractor


@ddt
class ITTestPubmedAbstractExtractor(unittest.TestCase):

    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_extract(self):
        # Arrange
        sut = PubmedAbstractExtractor()

        # Act
        actual = sut.extract_abstract_by_pubmedid(["25331875"])

        # Assert
        print(actual)

    def test_extract_exception(self):
        # Arrange
        sut = PubmedAbstractExtractor("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch2.fcgi")

        # Act
        with self.assertRaises(Exception) as context:
            sut.extract_abstract_by_pubmedid(["25331875"])

        # Assert
