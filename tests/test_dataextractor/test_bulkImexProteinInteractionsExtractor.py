from unittest import TestCase
from unittest.mock import MagicMock

from ddt import ddt, data, unpack

from dataextractors.BulkImexProteinInteractionsExtractor import BulkImexProteinInteractionsExtractor


@ddt
class TestBulkImexProteinInteractionsExtractor(TestCase):

    @data(["data/human_13_negative.xml"
              , "data/human_01.xml"])

    def test_extract_protein_interaction(self, list_files):
        # Arrange
        sut = BulkImexProteinInteractionsExtractor(['phosphorylation'])
        expected_total = 4
        sut.imexProteinInteractionsExtractor = MagicMock()
        sut.imexProteinInteractionsExtractor.get_protein_interactions.side_effect = lambda x: [{
            "isNegative": False
            , "participants": [{"uniprotid": "123", "alias": ["protien"]}]
            , "pubmedId": "56757"
            , "pubmedTitle": "Dummt"
            , "interactionType": "phosphorylation"
            , "interactionId": 1

        }, {
            "isNegative": True
            , "participants": [{"uniprotid": "345", "alias": ["protien"]}]
            , "pubmedId": "56757"
            , "pubmedTitle": "Dummt"
            , "interactionType": "phosphorylation"
            , "interactionId": 1

        }]

        # Act
        actual = list(sut.get_protein_interactions(list_files))

        # assert
        self.assertEqual(len(actual), expected_total)
