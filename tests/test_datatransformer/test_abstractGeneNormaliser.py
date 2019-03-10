from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from datatransformer.abstractGeneNormaliser import AbstractGeneNormaliser


class TestAbstractGeneNormaliser(TestCase):
    def test___call__(self):
        # Arrange
        annotations = [{'id': '19167335', 'type': 'a',
                        'text': 'PTP Protein tyrosine phosphatases MAPk ',
                        'annotations': [
                            {'start': '34', 'end': '38', 'name': 'MAPk', 'type': 'Gene', 'normalised_id': '10076'},

                            {'start': '0', 'end': '3', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10078'}
                        ]}]
        # Mock uniprot converter
        geneIdConverter = MagicMock()
        geneIdConverter.convert.side_effect = lambda x: {x: "Q{}".format(x)}

        sut = AbstractGeneNormaliser(annotations, geneIdConverter=geneIdConverter)

        data = pd.DataFrame([
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isValid": True,
             "participant1Id": "uni_10076",
             "participant1Alias": ["uni_10076_alias"],
             "participant2Id": "uni_5793",
             "participant2Alias": ["uni_5793_alias"],
             "pubmedId": "19167335", "pubmedTitle": "Q",
             "pubmedabstract": 'PTP Protein tyrosine phosphatases MAPk '}

        ])

        expected_data = [
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isValid": True,
             "participant1Id": "uni_10076",
             "participant1Alias": ["uni_10076_alias"],
             "participant2Id": "uni_5793",
             "participant2Alias": ["uni_5793_alias"],
             "pubmedId": "19167335", "pubmedTitle": "Q",
             "pubmedabstract": 'Q10078 Protein tyrosine phosphatases Q10076 '}

        ]
        # Act
        result_df = sut(data)

        for e in expected_data:
            self.assertEqual(e['pubmedabstract'],
                             result_df.query("pubmedId == '{}'".format(e['pubmedId']))['normalised_abstract'][0])
