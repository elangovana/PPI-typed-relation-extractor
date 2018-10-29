from unittest import TestCase

import pandas as pd

from datavisualiser.jsonPPIFlattenTransformer import IntactJsonPpiFlattenTransformer


class TestIntactJsonPpiFlattenTransformer(TestCase):

    def test_shouldtransform(self):
        data = pd.DataFrame([
            # Case 1 participant
            {"interactionId": "1",
             "interactionType": "phosphorylation",
             "isNegative": False,
             "participants": [{'uniprotid': 'P43405',
                               'alias': [['ksyk_human'], ['Tyrosine-protein kinase SYK'], ['SYK'],
                                         ['Spleen tyrosine kinase'], ['p72-Syk']]}],
             "pubmedId": "1", "pubmedTitle": "Q", "pubmedabstract": "TEST"}
            # Case 2 participants
            , {"interactionId": "2",
               "interactionType": "phosphorylation",
               "isNegative": False,
               "participants": [{'uniprotid': 'P43405',
                                 'alias': [['ksyk_human'], ['Tyrosine-protein kinase SYK'], ['SYK'],
                                           ['Spleen tyrosine kinase'], ['p72-Syk']]},
                                {'uniprotid': 'P434052',
                                 'alias': [['ksyk_human'], ['Tyrosine-protein kinase SYK'], ['SYK'],
                                           ['Spleen tyrosine kinase'], ['p72-Syk']]}
                                ],
               "pubmedId": "1", "pubmedTitle": "Q", "pubmedabstract": "TEST"}
            #Case 3 participants
            ,  {"interactionId": "3",
               "interactionType": "phosphorylation",
               "isNegative": False,
               "participants": [{'uniprotid': 'P43405',
                                 'alias': [['ksyk_human'], ['Tyrosine-protein kinase SYK'], ['SYK'],
                                           ['Spleen tyrosine kinase'], ['p72-Syk']]},
                                {'uniprotid': 'P434052',
                                 'alias': [['dest1']]}
                                , {'uniprotid': 'P434053',
                                 'alias': [['dest1'], ['dest2']]}
                                ],
               "pubmedId": "1", "pubmedTitle": "Q", "pubmedabstract": "TEST"}
        ])

        sut = IntactJsonPpiFlattenTransformer()

        actual = sut.transform(data)

        self.assertEquals(len(actual), 5)
