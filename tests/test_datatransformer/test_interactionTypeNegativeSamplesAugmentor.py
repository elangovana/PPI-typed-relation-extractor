import logging
import os
from logging.config import fileConfig
from unittest import TestCase

import pandas as pd

from datatransformer.interactionTypeNegativeSamplesAugmentor import InteractionTypeNegativeSamplesAugmentor


class TestInteractionTypeNegativeSamplesAugmentor(TestCase):

    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    def test_transform(self):
        # Arrange

        sut = InteractionTypeNegativeSamplesAugmentor()
        data = pd.DataFrame([
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isNegative": False,
             "participant1Id": "uni_10076",
             "participant1Alias": ["uni_10076_alias"],
             "participant2Id": "uni_5793",
             "participant2Alias": ["uni_5793_alias"],
             "pubmedId": "19167335", "pubmedTitle": "Q", "pubmedabstract": "TEST"}

            , {"interactionId": "2",
               "interactionType": "dephosphorylation",
               "isNegative": False,
               "participant1Id": "uni_26401",
               "participant1Alias": ["uni_26401_alias"],
               "participant2Id": "uni_26419",
               "participant2Alias": ["uni_26419_alias"],
               "pubmedId": "25260751", "pubmedTitle": "Q", "pubmedabstract": "TEST"}
            , {"interactionId": "3",
               "interactionType": "methylation",
               "isNegative": False,
               "participant1Id": "uni_264011",
               "participant1Alias": ["uni_264011_alias"],
               "participant2Id": "uni_26419",
               "participant2Alias": ["uni_26419_alias"],
               "pubmedId": "25260751", "pubmedTitle": "Q", "pubmedabstract": "TEST"}
        ])

        expected_fake = pd.DataFrame([
            {"interactionId": "random_guid_2_1",
             "interactionType": "dephosphorylation",
             "isNegative": True,
             "participant1Id": "uni_10076",
             "participant1Alias": ["uni_10076_alias"],
             "participant2Id": "uni_5793",
             "participant2Alias": ["uni_5793_alias"],
             "pubmedId": "19167335", "pubmedTitle": "Q", "pubmedabstract": "TEST"}

            , {"interactionId": "random_guid_1",
               "interactionType": "methylation",
               "isNegative": True,
               "participant1Id": "uni_10076",
               "participant1Alias": ["uni_10076_alias"],
               "participant2Id": "uni_5793",
               "participant2Alias": ["uni_5793_alias"],
               "pubmedId": "19167335", "pubmedTitle": "Q", "pubmedabstract": "TEST"}

            , {"interactionId": "2",
               "interactionType": "phosphorylation",
               "isNegative": True,
               "participant1Id": "uni_264011",
               "participant1Alias": ["uni_264011_alias"],
               "participant2Id": "uni_26419",
               "participant2Alias": ["uni_26419_alias"],
               "pubmedId": "25260751", "pubmedTitle": "Q", "pubmedabstract": "TEST"}

        ])

        # Act
        actual = sut.transform(data)

        # Assert
        sort_keys = ['pubmedId', "isNegative", "interactionType", 'participant1Id', 'participant2Id']
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
