from io import StringIO
from unittest import TestCase

from datatransformer.uniprotIdLocalDbMapper import UniprotIdLocalDbMapper


class TestUniprotIdLocalDbMapper(TestCase):

    def setUp(self):
        self._localdb = """P31946	UniProtKB-ID	1433B_HUMAN
        P31946	Gene_Name	YWHAB
        P31946	GI	4507949
        P31946	GI	377656702
        P31946	GI	67464628
        P31946	GI	1345590
        P31946	GI	1034625756
        P31946	GI	21328448
        P31946	GI	377656701
        P31946	GI	67464627
        P31946	GI	78101741
        P31947	Gene_Name	DEMO"""

    def test_convert_single_gene_name(self):
        # Arrange

        handle = StringIO(self._localdb)
        sut = UniprotIdLocalDbMapper(handle, "Gene_Name")
        input_uniprot_id = "P31946"
        expected_name = ["YWHAB"]

        # Act
        actual = sut.convert([input_uniprot_id])

        # Assert
        self.assertEqual(expected_name, actual[input_uniprot_id])

    def test_convert_multiple_target_names(self):
        # Arrange

        handle = StringIO(self._localdb)
        sut = UniprotIdLocalDbMapper(handle, "GI")
        input_uniprot_id = "P31946"
        expected = sorted([
            "4507949"
            , "377656702"
            , "67464628"
            , "1345590"
            , "1034625756"
            , "21328448"
            , "377656701"
            , "67464627"
            , "78101741"
        ])

        # Act
        actual = sut.convert([input_uniprot_id])

        # Assert
        self.assertEqual(expected, sorted(actual[input_uniprot_id]))

    def test_convert_multiple_uniprots(self):
        # Arrange

        handle = StringIO(self._localdb)
        sut = UniprotIdLocalDbMapper(handle, "Gene_Name")
        uniprot_a = "P31947"
        input_uniprot_ids = ["P31946", uniprot_a]
        expected_name = ["DEMO"]

        # Act
        actual = sut.convert(input_uniprot_ids)

        # Assert
        self.assertEqual(expected_name, actual[uniprot_a])
