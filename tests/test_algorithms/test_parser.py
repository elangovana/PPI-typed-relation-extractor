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

        actual = sut.transform_to_array(data, vocab=vocab)

        # Assert
        # Check the result length is accurate
        self.assertEqual(len(actual), len(data))
        # Check that each record contains the correct number of text columns
        for i in range(len(actual)):
            self.assertEqual(len(actual[i]), len(data[i]))

    def test_label_map(self):
        # Arrange
        labels = ["cat", "bat", "cat", "cat"]
        sut = Parser()

        # Act
        actual = sut.get_label_map(labels)

        self.assertSequenceEqual(['bat', 'cat'], actual.tolist())

    def test_encode_labels(self):
        # Arrange
        labels = ["cat", "bat", "cat", "cat"]
        sut = Parser()

        # Act
        actual = sut.encode_labels(labels, ['bat', 'cat'])

        self.assertSequenceEqual([1, 0, 1, 1], actual.tolist())
