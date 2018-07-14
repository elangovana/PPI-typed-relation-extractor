from io import BytesIO
from unittest import TestCase

import os
from ddt import ddt, data, unpack

from dataloader.dataPreprocessor import DataPreprocessor


@ddt
class TestDataPreprocessor(TestCase):

    @data(("data/human_13_negative.xml", 7))
    @unpack
    def test_Convert_to_json(self, xmlfile, expected_no_interactions):
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
