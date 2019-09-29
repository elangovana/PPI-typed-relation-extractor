from io import StringIO
from unittest import TestCase

from algorithms.collobert_embedding_formatter import CollobertEmbeddingFormatter


class TestCollobertEmbeddingFormatter(TestCase):
    def test_convert(self):
        # Arrange
        words_handle = StringIO("""the
NUMBER
""")
        words_embed = StringIO("""0.0 0.1
0.2 0.3
""")
        sut = CollobertEmbeddingFormatter(words_handle, words_embed)

        expected = StringIO("""0000000002 0000000002
the 0.0 0.1
NUMBER 0.2 0.3
""")

        actual = StringIO()

        # Act
        sut.convert(actual)

        # Assert
        self.assertEqual(expected.getvalue(), actual.getvalue())
