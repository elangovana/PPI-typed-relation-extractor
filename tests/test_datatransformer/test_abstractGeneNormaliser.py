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
                            {'start': '656', 'end': '659', 'name': 'Syk', 'type': 'Gene', 'normalised_id': '6850'},
                            {'start': '686', 'end': '695', 'name': 'caspase-1', 'type': 'Gene', 'normalised_id': '834'},
                            {'start': '726', 'end': '731', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
                            {'start': '790', 'end': '793', 'name': 'ASC', 'type': 'Gene', 'normalised_id': '29108'},
                            {'start': '802', 'end': '807', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
                            {'start': '834', 'end': '843', 'name': 'caspase-1', 'type': 'Gene', 'normalised_id': '834'},
                            {'start': '866', 'end': '869', 'name': 'Syk', 'type': 'Gene', 'normalised_id': '6850'},
                            {'start': '898', 'end': '901', 'name': 'ASC', 'type': 'Gene', 'normalised_id': '29108'},
                            {'start': '906', 'end': '911', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
                            {'start': '976', 'end': '979', 'name': 'Syk', 'type': 'Gene', 'normalised_id': '6850'},
                            {'start': '1217', 'end': '1220', 'name': 'Syk', 'type': 'Gene', 'normalised_id': '6850'},
                            {'start': '1230', 'end': '1235', 'name': 'NLRP3', 'type': 'Gene',
                             'normalised_id': '114548'},
                            {'start': '1298', 'end': '1301', 'name': 'ASC', 'type': 'Gene', 'normalised_id': '29108'},
                            {'start': '1324', 'end': '1327', 'name': 'Syk', 'type': 'Gene', 'normalised_id': '6850'},
                            {'start': '1368', 'end': '1373', 'name': 'NLRP3', 'type': 'Gene',
                             'normalised_id': '114548'},
                            {'start': '1408', 'end': '1413', 'name': 'NLRP3', 'type': 'Gene',
                             'normalised_id': '114548'},
                            {'start': '1093', 'end': '1096', 'name': 'ASC', 'type': 'Gene', 'normalised_id': '29108'},
                            {'start': '998', 'end': '1001', 'name': 'ASC', 'type': 'Gene', 'normalised_id': '29108'},
                            {'start': '755', 'end': '761', 'name': 'HEK293', 'type': 'Species',
                             'normalised_id': '9606'}]}

                       ]

        # Mock uniprot converter
        geneIdConverter = MagicMock()
        geneIdConverter.convert.side_effect = lambda x: {x: ["Q{}".format(x)]}

        sut = AbstractGeneNormaliser(annotations, geneIdConverter=geneIdConverter)

        data = pd.DataFrame([
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isValid": True,
             "participant1Id": "uni_10076",
             "participant1Alias": ["uni_10076_alias"],
             "participant2Id": "uni_5793",
             "participant2Alias": ["uni_5793_alias"],
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
             "participant2Alias": ["uni_5793_alias"],
             "pubmedId": "25605870", "pubmedTitle": "Q",
             "pubmedabstract": 'NLRP3 is the most crucial member of the NLR family, as it detects the existence of pathogen invasion and self-derived molecules associated with cellular damage. Several studies have reported that excessive NLRP3 inflammasome-mediated caspase-1 activation',

             "normalised_abstract": 'Q114548 is the most crucial member of the NLR family, as it detects the existence of pathogen invasion and self-derived molecules associated with cellular damage. Several studies have reported that excessive Q114548 inflammasome-mediated Q834 activation is a key factor in the development of diseases. Recent studies have reported that Q6850 is involved in pathogen-induced Q114548 inflammasome activation; however, the detailed mechanism linking Q6850 to Q114548 inflammasome remains unclear. In this study, we showed that Q6850 mediates Q114548 stimuli-induced processing of procaspase-1 and the consequent activation of Q834. Moreover, the kinase activity of Q6850 is required to potentiate Q834 activation in a reconstituted Q114548 inflammasome system in HEK293T cells. The adaptor protein Q29108 bridges Q114548 with the effector protein Q834. Herein, we find that Q6850 can associate directly with Q29108 and Q114548 by its kinase domain but interact indirectly with procaspase-1. Q6850 can phosphorylate Q29108 at Y146 and Y187 residues, and the phosphorylation of both residues is critical to enhance Q29108 oligomerization and the recruitment of procaspase-1. Together, our results reveal a new molecular pathway through which Q6850 promotes Q114548 inflammasome formation, resulting from the phosphorylation of Q29108. Thus, the control of Q6850 activity might be effective to modulate Q114548 inflammasome activation and treat Q114548-related immune diseases.'
             }

        ]
        # Act
        result_df = sut.transform(data)

        for e, r in zip(expected_data, result_df.values):
            self.assertEqual(e['normalised_abstract'], r[10])

    def test_transform_preferred(self):
        """
        When more than one matching uniprot for a gene name ncbi, it should select the one matching either of the participants in the dataframe
        :return:
        """
        # Arrange
        annotations = [{'id': '25605870', 'type': 'a',
                        'text': 'NLRP3 is the most crucial member of the NLR family',
                        'annotations': [
                            {'start': '0', 'end': '5', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'}]}
                       ]

        # Mock uniprot converter
        geneIdConverter = MagicMock()
        geneIdConverter.convert.side_effect = lambda x: {x: ["Q{}".format(x), "uni_10076"]}

        sut = AbstractGeneNormaliser(annotations, geneIdConverter=geneIdConverter)

        data = pd.DataFrame([
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isValid": True,
             "participant1Id": "uni_10076",
             "participant1Alias": ["uni_10076_alias"],
             "participant2Id": "uni_5793",
             "participant2Alias": ["uni_5793_alias"],
             "pubmedId": "25605870", "pubmedTitle": "Q",
             "pubmedabstract": 'NLRP3 is the most crucial member of the NLR family'}
        ])

        expected_data = [
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isValid": True,
             "participant1Id": "uni_10076",
             "participant1Alias": ["uni_10076_alias"],
             "participant2Id": "uni_5793",
             "participant2Alias": ["uni_5793_alias"],
             "pubmedId": "25605870", "pubmedTitle": "Q",
             "pubmedabstract": 'NLRP3 is the most crucial member of the NLR family',
             "normalised_abstract": 'uni_10076 is the most crucial member of the NLR family'}

        ]
        # Act
        result_df = sut.transform(data)

        for e, r in zip(expected_data, result_df.values):
            self.assertEqual(e['normalised_abstract'], r[10])

    def test_transform_preferred_alias(self):
        """
        When more than one matching uniprot for a gene name ncbi, it should select the one matching either of the participants in the dataframe
        :return:
        """
        # Arrange
        annotations = [{'id': '25605870', 'type': 'a',
                        'text': 'NLRP3 is the most crucial member of the NLR family',
                        'annotations': [
                            {'start': '0', 'end': '5', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'}]}
                       ]

        # Mock uniprot converter

        geneIdConverter = MagicMock()
        geneIdConverter.convert.side_effect = lambda x: {x: ["Q{}".format(x)]}

        sut = AbstractGeneNormaliser(annotations, geneIdConverter=geneIdConverter)

        data = pd.DataFrame([
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isValid": True,
             "participant1Id": "uni_10076",
             "participant1Alias": [["ABC"], ["nlrp3"]],
             "participant2Id": "uni_5793",
             "participant2Alias": [["uni_5793_alias"]],
             "pubmedId": "25605870", "pubmedTitle": "Q",
             "pubmedabstract": 'NLRP3 is the most crucial member of the NLR family'}
        ])

        expected_data = [
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isValid": True,
             "participant1Id": "uni_10076",
             "participant1Alias": [["ABC"], ["NLRP3"]],
             "participant2Id": "uni_5793",
             "participant2Alias": [["uni_5793_alias"]],
             "pubmedId": "25605870", "pubmedTitle": "Q",
             "pubmedabstract": 'NLRP3 is the most crucial member of the NLR family',
             "normalised_abstract": 'uni_10076 is the most crucial member of the NLR family'}

        ]
        # Act
        result_df = sut.transform(data)

        for e, r in zip(expected_data, result_df.values):
            self.assertEqual(e['normalised_abstract'], r[10])
