from unittest import TestCase

from ddt import ddt, data

from algorithms.Parser import Parser, UNKNOWN_WORD, EOS


@ddt
class TestParser(TestCase):
    @data((3), (-1))
    def test_transform_to_array(self, label_index):
        # Arrange
        vocab = {"This": 0, "good": 1, UNKNOWN_WORD: 3, EOS: 4}
        data = [(["This", "is", "good"], ["entity1"], ["entity2"], "0"),
                (["This", "is", "a" "hat,", "but", "not", "a", "cat"], ["mat"], ["cat"], "1")]
        sut = Parser()

        # Act

        actual = sut.transform_to_array(data, label_index=label_index, vocab=vocab)

        # Assert
        # Check the result length is accurate
        self.assertEqual(len(actual), len(data))
        # Check that each record contains the correct number of text columns
        for i in range(len(actual)):
            self.assertEqual(len(actual[i]), len(data[i]))

            # For each column exnure correct number of tokens
            for j in range(len(actual[i])):
                if j == label_index: continue
                # The result contains the vocab index and the end of sentence token
                self.assertEqual(len(actual[i][j]), len(data[i][j]) + 1)
