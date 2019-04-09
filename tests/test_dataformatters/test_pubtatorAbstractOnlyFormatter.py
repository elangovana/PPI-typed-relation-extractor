from io import StringIO
from unittest import TestCase

from dataformatters.pubtatorAbstractOnlyFormatter import PubtatorAbstractOnlyFormatter


class TestPubtatorAbstractOnlyFormatter(TestCase):
    def test___call__(self):
        """
    Verifies the formatting
            """
        # Arrange
        sut = PubtatorAbstractOnlyFormatter()
        data_iter = [{"pubmedid": "1233", "abstract": "This is a cat"},
                     {"pubmedid": "12133", "abstract": "This is a dog"}]
        output = StringIO()

        expected = "1233|a|This is a cat\n\n12133|a|This is a dog\n\n"

        # Act
        sut(data_iter, lambda x: x["pubmedid"], lambda x: x["abstract"], output)

        # Assert
        self.assertEqual(output.getvalue(), expected)

    def test___call__cleanuincode(self):
        """
    Verifies the formatting
            """
        # Arrange
        sut = PubtatorAbstractOnlyFormatter()
        data_iter = [{"pubmedid": "1233", "abstract": "This is a \u2808cat"},
                     {"pubmedid": "12133", "abstract": "This is a dog"}]
        output = StringIO()

        expected = "1233|a|This is a cat\n\n12133|a|This is a dog\n\n"

        # Act
        sut(data_iter, lambda x: x["pubmedid"], lambda x: x["abstract"], output)

        # Assert
        self.assertEqual(output.getvalue(), expected)
