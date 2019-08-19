import os
from io import StringIO
from unittest import TestCase

import pandas as pd

from datatransformer.AimedToDataFrame import AimedToDataFrame


class TestAimedToDataFrame(TestCase):
    def test___call__(self):
        # Arrange
        input_file = os.path.join(os.path.dirname(__file__), "data", "abstract_11780382")
        sut = AimedToDataFrame()
        expected_json = [{"docid": "abstract_11780382",
                          "line_no": 1,
                          "passage": "TI - Th1 / Th2 type cytokines in hepatitis B patients treated with interferon - alpha . "
                             , "participant1": "interferon - alpha"
                             , "participant2": None
                             , "isValid": False
                             ,

                          },

                         {"docid": "abstract_11780382",
                          "line_no": 2,
                          "passage": "Cytokines measurements during IFN - alpha treatment showed a trend to decreasing levels of IL - 4 at 4 , 12 , and 24 weeks ."
                             , "participant1": "IFN - alpha"
                             , "participant2": "IL - 4"
                             , "isValid": False

                          },
                         ]
        expected_df = pd.DataFrame(expected_json)

        # Act
        actual_df = sut(input_file)

        # Assert
        actual_list = actual_df.values.tolist()
        expected_list = expected_df.values.tolist()
        self.assertSequenceEqual(actual_list, expected_list)

    def test_parse(self):
        # Arrange
        sut = AimedToDataFrame()
        input_line = StringIO(
            "TI - Cloning of <p1  pair=1 >  <prot>  TRAP </prot>  </p1>  , a ligand for <p2  pair=1 >  <prot>  CD40 </prot>  </p2>  on human T cells .")
        expected_json = [{"docid": "abstract_11780382",
                          "line_no": 1,
                          "passage": "TI - Cloning of TRAP , a ligand for CD40 on human T cells ."
                             , "participant1": "TRAP"
                             , "participant2": "CD40"
                             , "isValid": True
                             ,

                          }
                         ]
        expected_df = pd.DataFrame(expected_json)

        # Act
        actual_df = sut.parse(input_line, "abstract_11780382")

        # Assert

        actual_list = actual_df.values.tolist()
        expected_list = expected_df.values.tolist()

        self.assertSequenceEqual(actual_list, expected_list)
