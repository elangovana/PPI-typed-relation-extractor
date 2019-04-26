from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from datatransformer.abstractGeneNormaliser import AbstractGeneNormaliser


class TestAbstractGeneNormaliser(TestCase):
    def test_transform(self):
        # Arrange
        annotations = [{'id': '25605870', 'type': 'a',
                        'text': 'NLRP3 is the most crucial member of the NLR family, as it detects the existence of pathogen invasion and self-derived molecules associated with cellular damage. Several studies have reported that excessive NLRP3 inflammasome-mediated caspase-1 activation is a key factor in the development of diseases. Recent studies have reported that Syk is involved in pathogen-induced NLRP3 inflammasome activation; however, the detailed mechanism linking Syk to NLRP3 inflammasome remains unclear. In this study, we showed that Syk mediates NLRP3 stimuli-induced processing of procaspase-1 and the consequent activation of caspase-1. Moreover, the kinase activity of Syk is required to potentiate caspase-1 activation in a reconstituted NLRP3 inflammasome system in HEK293T cells. The adaptor protein ASC bridges NLRP3 with the effector protein caspase-1. Herein, we find that Syk can associate directly with ASC and NLRP3 by its kinase domain but interact indirectly with procaspase-1. Syk can phosphorylate ASC at Y146 and Y187 residues, and the phosphorylation of both residues is critical to enhance ASC oligomerization and the recruitment of procaspase-1. Together, our results reveal a new molecular pathway through which Syk promotes NLRP3 inflammasome formation, resulting from the phosphorylation of ASC. Thus, the control of Syk activity might be effective to modulate NLRP3 inflammasome activation and treat NLRP3-related immune diseases.',
                        'annotations': [
                            {'start': '0', 'end': '5', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
                            {'start': '206', 'end': '211', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
                            {'start': '234', 'end': '243', 'name': 'caspase-1', 'type': 'Gene', 'normalised_id': '834'},
                            {'start': '337', 'end': '340', 'name': 'Syk', 'type': 'Gene', 'normalised_id': '6850'},
                            {'start': '373', 'end': '378', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
                            {'start': '444', 'end': '447', 'name': 'Syk', 'type': 'Gene', 'normalised_id': '6850'},
                            {'start': '451', 'end': '456', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
                            {'start': '517', 'end': '520', 'name': 'Syk', 'type': 'Gene', 'normalised_id': '6850'},
                            {'start': '530', 'end': '535', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
                            {'start': '612', 'end': '621', 'name': 'caspase-1', 'type': 'Gene', 'normalised_id': '834'},
                        ]}

                       ]

        # Mock uniprot converter
        mockTextNormaliser = MagicMock()
        mockTextNormaliser.return_value = "Normalised text.."

        sut = AbstractGeneNormaliser(annotations)
        sut.textGeneNormaliser = mockTextNormaliser

        data = pd.DataFrame([
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isValid": True,
             "participant1Id": "uni_10076",
             "participant1Alias": [["uni_10076_alias"]],
             "participant2Id": "uni_5793",
             "participant2Alias": [["uni_5793_alias"]],
             "pubmedId": "25605870", "pubmedTitle": "Q",
             "pubmedabstract": 'NLRP3 is the most crucial member of the NLR family, as it detects the existence of pathogen invasion and self-derived molecules associated with cellular damage. Several studies have reported that excessive NLRP3 inflammasome-mediated caspase-1 activation is a key factor in the development of diseases. Recent studies have reported that Syk is involved in pathogen-induced NLRP3 inflammasome activation; however, the detailed mechanism linking Syk to NLRP3 inflammasome remains unclear. In this study, we showed that Syk mediates NLRP3 stimuli-induced processing of procaspase-1 and the consequent activation of caspase-1. Moreover, the kinase activity of Syk is required to potentiate caspase-1 activation in a reconstituted NLRP3 inflammasome system in HEK293T cells. The adaptor protein ASC bridges NLRP3 with the effector protein caspase-1. Herein, we find that Syk can associate directly with ASC and NLRP3 by its kinase domain but interact indirectly with procaspase-1. Syk can phosphorylate ASC at Y146 and Y187 residues, and the phosphorylation of both residues is critical to enhance ASC oligomerization and the recruitment of procaspase-1. Together, our results reveal a new molecular pathway through which Syk promotes NLRP3 inflammasome formation, resulting from the phosphorylation of ASC. Thus, the control of Syk activity might be effective to modulate NLRP3 inflammasome activation and treat NLRP3-related immune diseases.'}

        ])

        expected_data = [
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isValid": True,
             "participant1Id": "uni_10076",
             "participant1Alias": ["uni_10076_alias"],
             "participant2Id": "uni_5793",
             "participant2Alias": [["uni_5793_alias"]],
             "pubmedId": "25605870", "pubmedTitle": "Q",
             "pubmedabstract": 'NLRP3 is the most crucial member of the NLR family, as it detects the existence of pathogen invasion and self-derived molecules associated with cellular damage. Several studies have reported that excessive NLRP3 inflammasome-mediated caspase-1 activation',

             "normalised_abstract": 'Normalised text..'}

        ]
        # Act
        result_df = sut.transform(data)

        for e, r in zip(expected_data, result_df.values):
            self.assertEqual(e['normalised_abstract'], r[10])
