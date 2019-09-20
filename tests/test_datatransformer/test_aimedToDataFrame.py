import os
from io import StringIO
from unittest import TestCase

import pandas as pd

from datatransformer.AimedToDataFrame import AimedToDataFrame


class TestAimedToDataFrame(TestCase):
    def test___call__(self):
        # Arrange
        input_file = os.path.join(os.path.dirname(__file__), "data", "aimed", "abstract_11780382")
        sut = AimedToDataFrame()
        expected_json = [
            # {"docid": "abstract_11780382",
            #                   "line_no": 1,
            #                   "passage": "TI - Th1 / Th2 type cytokines in hepatitis B patients treated with interferon - alpha . "
            #                      , "participant1": "interferon - alpha"
            #                      , "participant2": None
            #                      , "isValid": False
            #                      ,
            #
            #                   },

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
                             , "participant1": "CD40"
                             , "participant2": "TRAP"
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

    def test_parse_multiple_protiens(self):
        sut = AimedToDataFrame()
        input_line = StringIO(
            "In patients with a complete response to <prot>  IFN - alpha </prot>  , the levels of <prot>  IFN - gamma </prot>  were higher at 24 weeks following <prot>  IFN - alpha </prot>  treatment than that of pre - treatment ( P = 0.04 ) , and the levels of <prot>  IL - 4 </prot>  decreased markedly at 12 and 24 weeks ( P = 0.02 , 0.03 , respectively ) . mRNA expression positively correlated with the level of Th1 / Th2 type cytokines in the supernatant .")
        expected_json = [{"docid": "abstract_11780382",
                          "line_no": 1,
                          "passage": "In patients with a complete response to IFN - alpha , the levels of IFN - gamma were higher at 24 weeks following IFN - alpha treatment than that of pre - treatment ( P = 0.04 ) , and the levels of IL - 4 decreased markedly at 12 and 24 weeks ( P = 0.02 , 0.03 , respectively ) . mRNA expression positively correlated with the level of Th1 / Th2 type cytokines in the supernatant ."
                             , "participant1": "IFN - alpha"
                             , "participant2": "IFN - gamma"
                             , "isValid": False
                             ,

                          },
                         {"docid": "abstract_11780382",
                          "line_no": 1,
                          "passage": "In patients with a complete response to IFN - alpha , the levels of IFN - gamma were higher at 24 weeks following IFN - alpha treatment than that of pre - treatment ( P = 0.04 ) , and the levels of IL - 4 decreased markedly at 12 and 24 weeks ( P = 0.02 , 0.03 , respectively ) . mRNA expression positively correlated with the level of Th1 / Th2 type cytokines in the supernatant ."
                             , "participant1": "IFN - alpha"
                             , "participant2": "IL - 4"
                             , "isValid": False,

                          },
                         {"docid": "abstract_11780382",
                          "line_no": 1,
                          "passage": "In patients with a complete response to IFN - alpha , the levels of IFN - gamma were higher at 24 weeks following IFN - alpha treatment than that of pre - treatment ( P = 0.04 ) , and the levels of IL - 4 decreased markedly at 12 and 24 weeks ( P = 0.02 , 0.03 , respectively ) . mRNA expression positively correlated with the level of Th1 / Th2 type cytokines in the supernatant ."
                             , "participant1": "IFN - gamma"
                             , "participant2": "IL - 4"
                             , "isValid": False
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

    def test_parse_nested_relationships(self):
        sut = AimedToDataFrame()
        input_line = " <p1  pair=1 >  <p1  pair=2 >  <p1  pair=3 >  <prot> FGF - 7 </prot>  </p1>  </p1>  </p1>  recognizes one FGFR isoform known as the <p2  pair=1 >  <prot>  FGFR2 IIIb </prot>  </p2>  isoform or <p2  pair=2 >  <prot>  <prot>  keratinocyte growth factor </prot>  receptor </prot>  </p2>  ( <p2  pair=3 >  <prot>  KGFR </prot>  </p2>  ) , whereas <p1  pair=4 >  <p1  pair=5 >  <p1  pair=6 >  <p1  pair=7 >  <prot>  FGF - 2 </prot>  </p1>  </p1>  </p1>  </p1>  binds well to <p2  pair=4 >  <prot>  FGFR1 </prot>  </p2>  , <p2  pair=5 >  <prot>  FGFR2 </prot>  </p2>  , and <p2  pair=6 >  <prot>  FGFR4 </prot>  </p2>  but interacts poorly with <p2  pair=7 >  <prot>  KGFR </prot>  </p2>"
        expected_list = [frozenset(["FGF - 7", "FGFR2 IIIb"])
            , frozenset(["FGF - 7", "keratinocyte growth factor"])
            , frozenset(["FGF - 7", "keratinocyte growth factor receptor"])
            , frozenset(["FGF - 7", "KGFR"])
            , frozenset(["FGF - 2", "FGFR1"])
            , frozenset(["FGF - 2", "FGFR2"])
            , frozenset(["FGF - 2", "FGFR4"])
            , frozenset(["FGF - 2", "KGFR"])]
        expected_list.sort()

        # Act
        actual_list = list(sut._extract_relations(input_line))
        actual_list.sort()

        # Assert
        self.assertSequenceEqual(actual_list, expected_list)

    def test_parse_nested_relationships_2(self):
        sut = AimedToDataFrame()
        input_line = "Mutagenesis , recombinant protein expression , and physicochemical characterization were used to investigate the structural basis for the homodimerization and <p1  pair=4 >  <prot>  AKAP75 </prot>  </p1>  binding activities of <p1  pair=3 >  <p2  pair=3 >  <p2  pair=4 >  <prot>  RII beta </prot>  </p2>  </p2>  </p1>  "
        expected_list = [
            frozenset(["RII beta"])  # 3
            , frozenset(["AKAP75", "RII beta"])  # 4
        ]
        expected_list.sort()

        # Act
        actual_list = list(sut._extract_relations(input_line))
        actual_list.sort()

        # Assert
        self.assertSequenceEqual(actual_list, expected_list)

    def test_parse_nested_proteins(self):
        sut = AimedToDataFrame()
        input_line = "<p2  pair=2 >  <prot>  <prot>  keratinocyte growth factor </prot>  receptor </prot>  </p2> "
        expected_list = {"keratinocyte growth factor", "keratinocyte growth factor receptor"}

        # Act
        actual_list = sut._extract_proteins(input_line)

        # Assert
        self.assertSequenceEqual(actual_list, expected_list)
