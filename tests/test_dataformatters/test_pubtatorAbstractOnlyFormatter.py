import json
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

        expected = "".join([json.dumps({"text": "This is a cat", "sourcedb": "PubMed", "sourceid": "1233"}),
                            json.dumps({"text": "This is a dog", "sourcedb": "PubMed", "sourceid": "12133"})])

        # Act
        sut(data_iter, lambda x: x["pubmedid"], lambda x: x["abstract"], output)

        # Assert
        self.assertEqual(output.getvalue(), expected)
