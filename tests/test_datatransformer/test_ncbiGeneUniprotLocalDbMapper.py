import os
from unittest import TestCase

from ddt import ddt, data, unpack

from datatransformer.ncbiGeneUniprotLocalDbMapper import NcbiGeneUniprotLocalDbMapper


@ddt
class TestNcbiGeneUniprotLocalDbMapper(TestCase):

    @data(('10076', 1)
        , (['10076', '25930'], 2)
        , ('DUMMY', 0))
    @unpack
    def test_convert(self, geneid, expected_len):
        # Arrange
        localdb = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/sample_HUMAN_9606_idmapping.dat")
        with open(localdb, "r") as handle:
            sut = NcbiGeneUniprotLocalDbMapper(handle, "GeneID")

            # Act
            actual = sut.convert(geneid)

            # Assert
            self.assertEqual(len(actual), expected_len)
