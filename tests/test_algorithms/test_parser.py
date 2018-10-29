from unittest import TestCase

from ddt import ddt, data

from algorithms.Parser import Parser, UNKNOWN_WORD, EOS


@ddt
class TestParser(TestCase):


    def test_transform_to_array(self):
        # Arrange
        vocab = {"This": 0, "good": 1, UNKNOWN_WORD: 3, EOS: 4}
        data = [["This", "is", "good"], ["This", "is", "a" "hat,", "but", "not", "a", "cat"]]
        sut = Parser()

        # Act

        actual = sut.transform_to_array(data,  vocab=vocab)

        # Assert
        # Check the result length is accurate
        self.assertEqual(len(actual), len(data))
        # Check that each record contains the correct number of text columns
        for i in range(len(actual)):
            self.assertEqual(len(actual[i]), len(data[i]))
