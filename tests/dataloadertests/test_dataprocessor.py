from io import BytesIO
from unittest import TestCase

import os
from ddt import ddt, data

from dataloader.dataPreprocessor import DataPreprocessor


@ddt
class TestDataPreprocessor(TestCase):

    @data("data/human_13_negative.xml")
    def test_Convert_to_json(self, xmlfile):
        # Arrange
        fulXmlFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), xmlfile)

        outhandler = BytesIO()
        sut = DataPreprocessor()

        with open(fulXmlFilePath, "rb") as xmlHandler:
            # Act
            sut.transform(xmlHandler, outhandler)

        # Assert if output has any content
        outhandler.seek(0)
        self.assertEqual(len(outhandler.read(10)), 10)
