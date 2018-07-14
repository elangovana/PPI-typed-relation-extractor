from unittest import TestCase

import os
from ddt import ddt, data

from dataloader.elasticSearchLoader import ElasticSearchLoader

@ddt
class TestConvert_to_json(TestCase):

    @data("data/human_13_negative.xml")
    def test_Convert_to_json(self, xmlfile):
        #Arrange
        fulXmlFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), xmlfile)
        sut = ElasticSearchLoader()

        with open(fulXmlFilePath, "rb") as xmlHandler:
            #Act
            actual = sut.convert_to_json(xmlHandler)

        #Assert
        self.assertIsNotNone(actual)

