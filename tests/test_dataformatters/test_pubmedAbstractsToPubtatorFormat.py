import os
from io import StringIO
from unittest import TestCase
from unittest.mock import Mock

import pandas as pd

from dataformatters.pubmed_asbtracts_to_pubtator_format import PubmedAbstractsToPubtatorFormat


class TestPubmedAbstractsToPubtatorFormat(TestCase):

    def test___call__(self):
        # Arrange
        # Set up mock
        mock_pubtator_formatter = Mock()
        mock_pubtator_formatter.side_effect = lambda d, l, a, o: [o.write("{}\n".format(l(i))) for i in d]

        sut = PubmedAbstractsToPubtatorFormat()
        sut.pubtator_formatter = mock_pubtator_formatter

        # set up data
        expected = "30516273\n30516274\n"
        data = self._get_data()
        output_handle = StringIO()

        # Act
        sut(data, output_handle=output_handle)
        actual = output_handle.getvalue()

        # Assert
        self.assertEqual(actual, expected)

    def test_from_dataframe(self):
        # Arrange
        # Set up mock
        mock_pubtator_formatter = Mock()
        mock_pubtator_formatter.side_effect = lambda d, l, a, o: [o.write("{}\n".format(l(i))) for i in d]

        sut = PubmedAbstractsToPubtatorFormat()
        sut.pubtator_formatter = mock_pubtator_formatter

        # set up data
        expected = "30516273\n30516274\n"
        data = pd.DataFrame(self._get_data())
        output_handle = StringIO()

        # Act
        sut.from_dataframe(data, output_handle=output_handle)
        actual = output_handle.getvalue()

        # Assert
        self.assertEqual(actual, expected)

    def test_read_json(self):
        # Arrange
        # Set up mock
        mock_pubtator_formatter = Mock()
        mock_pubtator_formatter.side_effect = lambda d, l, a, o: [o.write("{}\n".format(l(i))) for i in d]

        sut = PubmedAbstractsToPubtatorFormat()
        sut.pubtator_formatter = mock_pubtator_formatter

        # set up data
        expected = "30516273\n30516274\n"
        input_file = os.path.join(os.path.dirname(__file__), "data_pubmedabstract.json")
        output_handle = StringIO()

        # Act
        sut.read_json(input_file, output_handle=output_handle)
        actual = output_handle.getvalue()

        # Assert
        self.assertEqual(actual, expected)

    def _get_data(self):
        return [
            {"pubmed_id": "30516273",
             "article_title": "Temporal clustering of extreme climate events drives a regime shift in rocky intertidal biofilms.",
             "article_abstract": "Research on regime shifts has focused primarily on how changes in the intensity ..",
             "pub_date": {
                 "year": "2018",
                 "month": "Dec",
                 "day": "05"
             }
             },
            {"pubmed_id": "30516274",
             "article_title": "Multiple drivers of contrasting diversity-invasibility relationships at fine spatial grains.",
             "article_abstract": "The diversity-invasibility hypothesis and ecological theory..",
             "pub_date": {
                 "year": "2018",
                 "month": "Dec",
                 "day": "05"
             }
             }
        ]
