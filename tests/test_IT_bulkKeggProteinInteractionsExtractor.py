from logging.config import fileConfig
from unittest import TestCase

import os
from ddt import ddt, data, unpack

from BulkKeggProteinInteractionsExtractor import BulkKeggProteinInteractionsExtractor
from KeggProteinInteractionsExtractor import KeggProteinInteractionsExtractor


@ddt
class TestBulkKeggProteinInteractionsExtractor(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data((["path:hsa04110", "path:ko05215"], 500))
    @unpack
    def test_extract(self, pathway_list, expected_no_min_records):
        # Arrange
        # kegg
        kegg = KeggProteinInteractionsExtractor()
        sut = BulkKeggProteinInteractionsExtractor(kegg)

        # act
        actual = sut.extract(pathway_list)

        # Assert
        self.assertGreater(expected_no_min_records, len(actual))


    def test_get_all(self):
        # Arrange
        # kegg
        kegg = KeggProteinInteractionsExtractor()
        sut = BulkKeggProteinInteractionsExtractor(kegg)

        # act
        actual = sut.extract_all()

# assert
        self.assertTrue(not actual is None)

