from unittest import TestCase

from ddt import ddt, data, unpack

from datatransformer.ncbiGeneUniprotMapper import NcbiGeneUniprotMapper


@ddt
class TestNcbiGeneUniprotMapper(TestCase):

    @data(('10076', 1)
        , (['10076', '25930'], 2)
        , ('DUMMY', 0))
    @unpack
    def test_convert(self, geneid, expected_len):
        # Arrange
        sut = NcbiGeneUniprotMapper()

        # Act
        actual = sut.convert(geneid)
        print(actual)

        # Assert
        self.assertEqual(len(actual), expected_len)
