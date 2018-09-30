from io import StringIO
from unittest import TestCase

from dataformatters.fastTextFormatter import FastTextFormatter


class TestFastTextFormatter(TestCase):
    def test_format(self):
        """
Verifies the formatting
        """
        # Arrange
        sut = FastTextFormatter()
        data_iter = [{"label": "cat", "text": "This is a cat"}, {"label": "dog", "text": "This is a dog"}]
        output = StringIO()

        expected = "\n".join(["__label__cat This is a cat", "__label__dog This is a dog\n"])

        # Act
        sut(data_iter, lambda x: x["label"], lambda x: x["text"], output)

        # Assert
        self.assertEqual(output.getvalue(), expected)
