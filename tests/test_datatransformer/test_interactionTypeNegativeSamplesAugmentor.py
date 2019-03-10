import logging
import os
from logging.config import fileConfig
from unittest import TestCase

import pandas as pd

from datatransformer.interactionTypeNegativeSamplesAugmentor import InteractionTypeNegativeSamplesAugmentor


class TestInteractionTypeNegativeSamplesAugmentor(TestCase):

    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_transform_large_negatives_per_pubmed(self):
        # Arrange

        sut = InteractionTypeNegativeSamplesAugmentor(max_negative_per_pubmed=1000)
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
               "interactionType": "dephosphorylation",
               "isValid": True,
               "participant1Id": "uni_26401",
               "participant1Alias": ["uni_26401_alias"],
               "participant2Id": "uni_26419",
               "participant2Alias": ["uni_26419_alias"],
               "pubmedId": "25260751", "pubmedTitle": "Q", "pubmedabstract": "TEST"}
            , {"interactionId": "3",
               "interactionType": "methylation",
               "isValid": True,
               "participant1Id": "uni_264011",
               "participant1Alias": ["uni_264011_alias"],
               "participant2Id": "uni_26419",
               "participant2Alias": ["uni_26419_alias"],
               "pubmedId": "25260751", "pubmedTitle": "Q", "pubmedabstract": "TEST"}
        ])

        expected_fake = pd.DataFrame([
            {"interactionId": "random_guid_2_1",
             "interactionType": "dephosphorylation",
             "isValid": False,
             "participant1Id": "uni_10076",
             "participant1Alias": ["uni_10076_alias"],
             "participant2Id": "uni_5793",
             "participant2Alias": ["uni_5793_alias"],
             "pubmedId": "19167335", "pubmedTitle": "Q", "pubmedabstract": "TEST"}

            , {"interactionId": "random_guid_1",
               "interactionType": "methylation",
               "isValid": False,
               "participant1Id": "uni_10076",
               "participant1Alias": ["uni_10076_alias"],
               "participant2Id": "uni_5793",
               "participant2Alias": ["uni_5793_alias"],
               "pubmedId": "19167335", "pubmedTitle": "Q", "pubmedabstract": "TEST"}

            , {"interactionId": "2",
               "interactionType": "phosphorylation",
               "isValid": False,
               "participant1Id": "uni_264011",
               "participant1Alias": ["uni_264011_alias"],
               "participant2Id": "uni_26419",
               "participant2Alias": ["uni_26419_alias"],
               "pubmedId": "25260751", "pubmedTitle": "Q", "pubmedabstract": "TEST"}

        ])

        # Act
        actual = sut.transform(data)

        # Assert
        sort_keys = ['pubmedId', "isValid", "interactionType", 'participant1Id', 'participant2Id']
        # dropping interaction_id because they are auto generated guids and may not remain consistent across tests.
        actual = actual.drop(["interactionId"], axis=1).sort_values(by=sort_keys)
        expected = pd.DataFrame(columns=data.columns)
        expected = expected.append(expected_fake)
        expected = expected.append(data)
        expected = expected.drop(["interactionId"], axis=1).sort_values(by=sort_keys)

        # logging
        if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
            logger = logging.getLogger(__name__)
            for a, e in zip(actual.values, expected.values):
                logger.debug("Actual {}".format(a))
                logger.debug("Expect {}\n".format(e))

        self.assertEqual(len(actual.values), len(expected.values))
        for a, e in zip(actual.values, expected.values):
            for ac, ec in zip(a, e):
                self.assertEqual(ac, ec)

    def test_transform_one_negative_per_pubmed(self):
        # Arrange

        max_fake_per_pubmed = 1
        sut = InteractionTypeNegativeSamplesAugmentor(max_negative_per_pubmed=max_fake_per_pubmed)
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
               "interactionType": "dephosphorylation",
               "isValid": True,
               "participant1Id": "uni_26401",
               "participant1Alias": ["uni_26401_alias"],
               "participant2Id": "uni_26419",
               "participant2Alias": ["uni_26419_alias"],
               "pubmedId": "25260751", "pubmedTitle": "Q", "pubmedabstract": "TEST"}
            , {"interactionId": "3",
               "interactionType": "methylation",
               "isValid": True,
               "participant1Id": "uni_264011",
               "participant1Alias": ["uni_264011_alias"],
               "participant2Id": "uni_26419",
               "participant2Alias": ["uni_26419_alias"],
               "pubmedId": "25260751", "pubmedTitle": "Q", "pubmedabstract": "TEST"}
        ])

        # Act
        actual = sut.transform(data)

        # logging
        if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
            logger = logging.getLogger(__name__)
            for a in actual.values:
                logger.debug("Actual {}".format(a))

        # Assert that you have utmost one fake record per pubmed
        for p in data.pubmedId.unique():
            a_match_records = actual.query("pubmedId == '{}'".format(p))
            a_match_records_fake = a_match_records.query('isValid == False')
            a_match_records_valid = a_match_records.query('isValid')
            e_valid = data.query("pubmedId == '{}'".format(p))
            self.assertEqual(e_valid.shape, a_match_records_valid.shape,
                             "The number of postive class should match the input")
            self.assertEqual(max_fake_per_pubmed, a_match_records_fake.shape[0],
                             "The number of fake classs do not match the expected")
