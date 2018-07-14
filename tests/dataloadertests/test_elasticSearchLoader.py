import json
from unittest import TestCase

import os
from ddt import ddt, data, unpack

from dataloader.elasticSearchLoader import ElasticSearchLoader

@ddt
class TestConvert_to_json(TestCase):

    @data(("<x>a</x>", "{\"x\": \"a\"}"))
    @unpack
    def test_Convert_to_json(self, xml, expected):
        #Arrange

        sut = ElasticSearchLoader()

        #Act
        actual = sut.convert_to_json(xml)

        #Assert
        self.assertEqual(json.loads(actual), json.loads(expected))

