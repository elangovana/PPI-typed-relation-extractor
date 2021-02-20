import os
from logging.config import fileConfig
from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from datatransformer.gnormplusNegativeSamplesAugmentor import GnormplusNegativeSamplesAugmentor


class TestGnormplusNegativeSamplesAugmentor(TestCase):

    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_transform(self):
        # Arrange

        annotations = [{'id': '19167335', 'type': 'a',
                        'text': 'Protein tyrosine phosphatases (PTPs) play a critical role in regulating cellular functions by selectively dephosphorylating their substrates. Here we present 22 human PTP crystal structures that, together with prior structural knowledge, enable a comprehensive analysis of the classical PTP family. Despite their largely conserved fold, surface properties of PTPs are strikingly diverse. A potential secondary substrate-binding pocket is frequently found in phosphatases, and this has implications for both substrate recognition and development of selective inhibitors. Structural comparison identified four diverse catalytic loop (WPD) conformations and suggested a mechanism for loop closure. Enzymatic assays revealed vast differences in PTP catalytic activity and identified PTPD1, PTPD2, and HDPTP as catalytically inert protein phosphatases. We propose a "head-to-toe" dimerization model for RPTPgamma/zeta that is distinct from the "inhibitory wedge" model and that provides a molecular basis for inhibitory regulation. This phosphatome resource gives an expanded insight into intrafamily PTP diversity, catalytic activity, substrate recognition, and autoregulatory self-association.',
                        'annotations': [
                            {'start': '167', 'end': '170', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
                            {'start': '898', 'end': '907', 'name': 'RPTPgamma', 'type': 'Gene',
                             'normalised_id': '5793'},
                            {'start': '741', 'end': '744', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
                            {'start': '161', 'end': '166', 'name': 'human', 'type': 'Species', 'normalised_id': '9606'},
                            {'start': '1096', 'end': '1099', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'}]}
            , {'id': '25260751', 'type': 'a',
               'text': 'Unlike the other MAP3Ks, MEKK1 (encoded by Map3k1) contains a PHD motif. To understand the role of this motif, we have created a knockin mutant of mouse Map3k1 (Map3k1(m) (PHD)) with an inactive PHD motif. Map3k1(m) (PHD) ES cells demonstrate that the MEKK1 PHD controls p38 and JNK activation during TGF-b, EGF and microtubule disruption signalling, but does not affect MAPK responses to hyperosmotic stress. Protein microarray profiling identified the adaptor TAB1 as a PHD substrate, and TGF-b- or EGF-stimulated Map3k1(m) (PHD) ES cells exhibit defective non-canonical ubiquitination of MEKK1 and TAB1. The MEKK1 PHD binds and mediates the transfer of Lys63-linked poly-Ub, using the conjugating enzyme UBE2N, onto TAB1 to regulate TAK1 and MAPK activation by TGF-b and EGF. Both the MEKK1 PHD and TAB1 are critical for ES-cell differentiation and tumourigenesis. Map3k1(m) (PHD) (/+) mice exhibit aberrant cardiac tissue, B-cell development, testis and T-cell signalling. ',
               'annotations': [{'start': '25', 'end': '30', 'name': 'MEKK1', 'type': 'Gene', 'normalised_id': '26401'},
                               {'start': '43', 'end': '49', 'name': 'Map3k1', 'type': 'Gene', 'normalised_id': '26401'},
                               {'start': '153', 'end': '159', 'name': 'Map3k1', 'type': 'Gene',
                                'normalised_id': '26401'},
                               {'start': '271', 'end': '274', 'name': 'p38', 'type': 'Gene', 'normalised_id': '26416'},
                               {'start': '279', 'end': '282', 'name': 'JNK', 'type': 'Gene', 'normalised_id': '26419'},
                               {'start': '889', 'end': '893', 'name': 'mice', 'type': 'Species',
                                'normalised_id': '10090'}]}

                       ]
        # Mock uniprot converter
        geneIdConverter = MagicMock()
        geneIdConverter.convert.side_effect = lambda x: {x: ["uni_{}".format(x)]}

        sut = GnormplusNegativeSamplesAugmentor(annotations, geneIdConverter)
        data = pd.DataFrame([
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isValid": True,
             "participant1Id": "uni_10076",
             "participant1Alias": ["uni_10076_alias"],
             "participant2Id": "uni_5793",
             "participant2Alias": ["uni_5793_alias"],
             "pubmedId": "19167335", "pubmedTitle": "Q", "pubmedabstract": "TEST"}

            , {"interactionId": "2",
               "interactionType": "phosphorylation",
               "isValid": True,
               "participant1Id": "uni_26401",
               "participant1Alias": ["Map3k1", "MEKK1"],
               "participant2Id": "uni_26419",
               "participant2Alias": ["JNK"],
               "pubmedId": "25260751", "pubmedTitle": "Q", "pubmedabstract": "TEST"}
        ])

        expected_fake = pd.DataFrame([
            {
                "interactionId": '2_0f63748e-10d6-4943-88a1-88c9a7778267_fake_annot',
                "interactionType": 'phosphorylation',
                "isValid": False, "participant1Id": 'uni_26416', "participant2Id": 'uni_26419',
                "pubmedId": '25260751', "pubmedTitle": 'Q', "pubmedabstract": 'TEST', "participant2Alias": ['JNK'],
                "participant1Alias": ['p38']
            }
            ,
            {"interactionId": '2_52039565-0320-4867-b8ee-fded5ef2ef36_fake_annot',
             "interactionType": 'phosphorylation',
             "isValid": False, "participant1Id": 'uni_26401', "participant2Id": 'uni_26416', "pubmedId": '25260751',
             "pubmedTitle": 'Q', "pubmedabstract": 'TEST', "participant1Alias": ['MEKK1', 'Map3k1'],
             "participant2Alias": ['p38']}
        ])

        sort_keys = ['pubmedId', 'isValid', 'participant1Id', 'participant2Id']
        columns = list(expected_fake.columns)
        columns.remove("interactionId")

        # format expected..
        expected = expected_fake
        expected = expected.append(data)

        expected = expected.sort_values(by=sort_keys)[columns]

        # Act
        actual = sut.transform(data)

        # Assert
        actual = actual.sort_values(by=sort_keys)[columns]
        self.assertEqual(len(actual.values), len(expected.values),
                         "Expected and actual does not match \n{}\n{}".format(expected.values, actual.values))

        for a, e in zip(actual.values, expected.values):
            self.assertSequenceEqual(a.tolist(), e.tolist())
