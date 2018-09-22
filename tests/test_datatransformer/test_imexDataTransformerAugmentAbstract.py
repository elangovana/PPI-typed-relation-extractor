from unittest import TestCase
from unittest.mock import MagicMock

from datatransformer.ImexDataTransformerAugmentAbstract import ImexDataTransformerAugmentAbstract


class TestImexDataTransformerAugmentAbstract(TestCase):

    def test_should_transform(self):
        # Arrange
        sut = ImexDataTransformerAugmentAbstract()
        key_pubmedid = "pubmedid"
        key_pubmedabstract = "pubmedabstract"
        input = [{key_pubmedid: "1234"}]
        expected = [{key_pubmedid: "1234", key_pubmedabstract: "Dummy Abstract"}]

        # mock
        sut.pubmed_extractor = MagicMock()
        sut.pubmed_extractor.extract_abstract_by_pubmedid.return_value = [{
            "id": "1234"
            , "title": "Dummy"
            , 'abstract': "Dummy Abstract"

        }]

        # Act
        actual = list(sut.transform(input, id_key=key_pubmedid, abstract_key=key_pubmedabstract))

        # Assert
        self.assertSequenceEqual(actual, expected)
