from unittest import TestCase

from algorithms.VocabMerge import VocabMerger


class TestVocabMerger(TestCase):
    def test___call__vocab_same(self):
        """
        Test case: case vocab is the same
        """
        # Arrange
        input1 = {"w": 0}
        input2 = {"w": 0}
        sut = VocabMerger()

        expected_vocab = {"w": 0}

        # Act
        actual_vocab = sut(input1, input2)

        # Assert
        self.assertEqual(expected_vocab, actual_vocab)

    def test___call__vocab_different(self):
        """
        Test case: case vocab is different
        """
        # Arrange
        input1 = {"w": 0}
        input2 = {"b": 0}
        sut = VocabMerger()

        expected_vocab = {"w": 0, "b": 1}

        # Act
        actual_vocab = sut(input1, input2)

        # Assert
        self.assertEqual(expected_vocab, actual_vocab)

    def test___call__vocab_partially_different(self):
        """
        Test case: case vocab is partially different
        """
        # Arrange
        input1 = {"w": 0}
        input2 = {"b": 0, "w": 1}
        sut = VocabMerger()

        expected_vocab = {"w": 0, "b": 1}

        # Act
        actual_vocab = sut(input1, input2)

        # Assert
        self.assertEqual(expected_vocab, actual_vocab)
