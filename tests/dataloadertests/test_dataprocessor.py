import json
import logging
from io import BytesIO, StringIO
from logging.config import fileConfig
from unittest import TestCase
from unittest.mock import MagicMock

import os
from xml.etree import ElementTree

from ddt import ddt, data, unpack

from dataloader.PubmedAbstractExtractor import PubmedAbstractExtractor
from dataloader.dataPreprocessor import DataPreprocessor
from pathlib import Path
from unittest.mock import MagicMock


@ddt
class TestDataPreprocessor(TestCase):
    def setUp(self):
        self._logger = logging.getLogger(__name__)
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data(("data/human_13_negative.xml", 7))
    @unpack
    def test_transform(self, xmlfile, expected_no_interactions):
        # Arrange
        fulXmlFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), xmlfile)
        sut = DataPreprocessor()

        with open(fulXmlFilePath, "rb") as xmlHandler:
            # Act
            actual = []
            for item in sut.transform(xmlHandler):
                actual.append(item)

        # Assert if output has any content
        self.assertEqual(len(actual), expected_no_interactions)

    @data("data/human_transformed_01_part1.xml")
    def test_adddata(self, xmlfile):
        # Arrange
        fulXmlFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), xmlfile)
        contents = Path(fulXmlFilePath).read_text()
        content_handle = StringIO(contents)

        sut = DataPreprocessor()
        mock_extractor = PubmedAbstractExtractor()
        abstract_dummy = "This is a dummy extract"
        mock_extractor.extract_abstract_by_pubmedid = MagicMock(return_value=[abstract_dummy])
        sut.pubmed_extractor=mock_extractor


        # Act
        actual = sut.adddata(content_handle)

        # Assert if output has any content

        self.assertTrue(abstract_dummy in actual)

    @data(("<x>a</x>", "{\"x\": \"a\"}"))
    @unpack
    def test_Convert_to_json(self, xml, expected):
        # Arrange
        sut = DataPreprocessor()

        # Act
        actual = sut.convert_to_json(xml)

        # Assert
        self.assertEqual(json.loads(actual), json.loads(expected))
