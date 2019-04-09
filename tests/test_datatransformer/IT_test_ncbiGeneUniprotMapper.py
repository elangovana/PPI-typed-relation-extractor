from unittest import TestCase

from ddt import ddt, data, unpack

from datatransformer.ncbiGeneUniprotMapper import NcbiGeneUniprotMapper


@ddt
class TestNcbiGeneUniprotMapper(TestCase):

    @data(('10076', 1)
        , ('6850', 1)
        , (['10076', '25930'], 2)
        , ('DUMMY', 0))
    @unpack
    def test_convert(self, geneid, expected_len):
        # Arrange
        sut = NcbiGeneUniprotMapper()

        # Act
        actual = sut.convert(geneid)


        # Assert
        self.assertEqual(len(actual), expected_len)

    @data(('6850', 3))
    @unpack
    def test_convert_more_than_one_match(self, geneid, expected_len):
        # Arrange
        sut = NcbiGeneUniprotMapper()

        # Act
        actual = sut.convert(geneid)

        # Assert
        self.assertEqual(len(actual[geneid]), expected_len)
