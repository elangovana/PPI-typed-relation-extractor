from io import StringIO
from unittest import TestCase

import pandas as pd

from datatransformer.AimedXmlToDataFramePreprocessed import AimedXmlToDataFramePreprocessed


class TestAimedXmlToDataFramePreprocessed(TestCase):
    def test___call__no_relation(self):
        # Arrange
        xml = """
<corpus source="AIMed">
    <document id="AIMed.d0">
        <sentence id="AIMed.d0.s0" text="Th1/Th2 type cytokines in hepatitis B patients treated with interferon-alpha."
                  seqId="s0">
            <entity id="AIMed.d0.s0.e0" charOffset="60-75" type="protein" text="interferon-alpha" seqId="e0"/>
        </sentence>
    </document>
</corpus>
        """

        xml_handle = StringIO(xml)

        sut = AimedXmlToDataFramePreprocessed()

        expected_json = []

        expected_df = pd.DataFrame(expected_json)

        # Act
        dataframe = sut(xml_handle)

        # Assert
        self.assertSequenceEqual(dataframe.values.tolist(), expected_df.values.tolist())

    def test___call__no_relation_2entities(self):
        # Arrange
        xml = """
<corpus source="AIMed">
    <document id="AIMed.d0">
        <sentence id="AIMed.d0.s5" text="Cytokines measurements during IFN-alpha treatment showed a trend to decreasing levels of IL-4 at 4, 12, and 24 weeks." seqId="s5">
          <entity id="AIMed.d0.s5.e0" charOffset="30-38" type="protein" text="IFN-alpha" seqId="e5"/>
          <entity id="AIMed.d0.s5.e1" charOffset="89-92" type="protein" text="IL-4" seqId="e6"/>
        </sentence>
    </document>
</corpus>
        """

        xml_handle = StringIO(xml)

        sut = AimedXmlToDataFramePreprocessed()

        expected_json = [{"docid": "AIMed.d0"
                             ,
                          "passage": "Cytokines measurements during IFN-alpha treatment showed a trend to decreasing levels of IL-4 at 4, 12, and 24 weeks."
                             , "passageid": "AIMed.d0.s5"
                             , "participant1": "IFN-alpha"
                             , "participant1_loc": "30-38"

                             , "participant2": "IL-4"
                             , "participant2_loc": "89-92"

                             , "isValid": False

                          }]

        expected_df = pd.DataFrame(expected_json)

        # Act
        dataframe = sut(xml_handle)

        # Assert
        self.assertSequenceEqual(dataframe.values.tolist(), expected_df.values.tolist())

    def test___call__one_relation_relation(self):
        # Arrange
        xml = """
<corpus source="AIMed">
    <document id="AIMed.d0">
          <sentence id="AIMed.d30.s255" text="We also found another armadillo-protein, p0071, interacted with PS1." seqId="s255">
              <entity id="AIMed.d30.s255.e1" charOffset="41-45" type="protein" text="p0071" seqId="e409"/>
              <entity id="AIMed.d30.s255.e2" charOffset="64-66" type="protein" text="PS1" seqId="e411"/>
              <interaction id="AIMed.d30.s255.i0" e1="AIMed.d30.s255.e1" e2="AIMed.d30.s255.e2" type="None" directed="false" seqId="i12"/>
        </sentence>
    </document>
</corpus>
        """

        xml_handle = StringIO(xml)

        sut = AimedXmlToDataFramePreprocessed()

        expected_json = [{"docid": "AIMed.d0"
                             ,
                          "passage": "We also found another armadillo-protein, p0071, interacted with PS1."
                             , "passageid": "AIMed.d30.s255"
                             , "participant1": "p0071"
                             , "participant1_loc": "41-45"
                             , "participant2": "PS1"
                             , "participant2_loc": "64-66"

                             , "isValid": True

                          }]

        expected_df = pd.DataFrame(expected_json)

        # Act
        dataframe = sut(xml_handle)

        # Assert
        self.assertSequenceEqual(dataframe.values.tolist(), expected_df.values.tolist())

    def test___call__pos_neg_relation(self):
        # Arrange
        xml = """
<corpus source="AIMed">
    <document id="AIMed.d0">
          <sentence id="AIMed.d30.s255" text="We also found another armadillo-protein, p0071, interacted with PS1." seqId="s255">
                    <entity id="AIMed.d30.s255.e0" charOffset="22-38" type="protein" text="armadillo-protein" seqId="e408"/>
                    <entity id="AIMed.d30.s255.e1" charOffset="41-45" type="protein" text="p0071" seqId="e409"/>
              <entity id="AIMed.d30.s255.e2" charOffset="64-66" type="protein" text="PS1" seqId="e411"/>
              <interaction id="AIMed.d30.s255.i0" e1="AIMed.d30.s255.e1" e2="AIMed.d30.s255.e2" type="None" directed="false" seqId="i12"/>
        </sentence>
    </document>
</corpus>
        """

        xml_handle = StringIO(xml)

        sut = AimedXmlToDataFramePreprocessed()

        expected_json = [{"docid": "AIMed.d0"
                             ,
                          "passage": "We also found another PROTEIN , p0071, interacted with PS1."
                             , "passageid": "AIMed.d30.s255"
                             , "participant1": "p0071"
                             , "participant1_loc": "41-45"
                             , "participant2": "PS1"
                             , "participant2_loc": "64-66"

                             , "isValid": True

                          },
                         {"docid": "AIMed.d0"
                             , "passage": "We also found another armadillo-protein, p0071, interacted with PROTEIN ."
                             , "passageid": "AIMed.d30.s255"
                             , "participant1_loc": "22-38"

                             , "participant1": "armadillo-protein"
                             , "participant2": "p0071"
                             , "participant2_loc": "41-45"

                             , "isValid": False

                          },
                         {"docid": "AIMed.d0"
                             , "passage": "We also found another armadillo-protein, PROTEIN , interacted with PS1."
                             , "passageid": "AIMed.d30.s255"
                             , "participant1": "armadillo-protein"
                             , "participant1_loc": "22-38"
                             , "participant2": "PS1"
                             , "participant2_loc": "64-66"

                             , "isValid": False

                          }
                         ]

        sort_cols = ["isValid", "participant1_loc", "participant2_loc"]
        expected_df = pd.DataFrame(expected_json).sort_values(by=sort_cols)

        # Act
        dataframe = sut(xml_handle).sort_values(by=sort_cols)

        # Assert
        self.assertSequenceEqual(dataframe.values.tolist(), expected_df.values.tolist())

    def test___call__multiple_sentences(self):
        # Arrange
        xml = """
<corpus source="AIMed">
     <document id="AIMed.d35">
        <sentence id="AIMed.d35.s291" text="Endogenous presenilin 1 redistributes to the surface of lamellipodia upon adhesion of Jurkat cells to a collagen matrix." seqId="s291">
          <entity id="AIMed.d35.s291.e0" charOffset="11-22" type="protein" text="presenilin 1" seqId="e569"/>
        </sentence>
        <sentence id="AIMed.d35.s292" text="Most familial early-onset Alzheimer's disease cases are caused by mutations in the presenilin 1 (PS1) gene." seqId="s292">
          <entity id="AIMed.d35.s292.e0" charOffset="83-94" type="protein" text="presenilin 1" seqId="e570"/>
          <entity id="AIMed.d35.s292.e1" charOffset="97-99" type="protein" text="PS1" seqId="e571"/>
        </sentence>
    </document>
</corpus>
        """

        xml_handle = StringIO(xml)

        sut = AimedXmlToDataFramePreprocessed()

        expected_count = 1

        # Act
        dataframe = sut(xml_handle)

        # Assert
        self.assertEqual(dataframe.shape[0], expected_count)

    def test__normalise_protien_names(self):
        # Arrange
        text = "A B C D"

        offsets = ["6-7", "0-1", "4-5"]

        expected = " PROTEIN B PROTEIN PROTEIN "

        sut = AimedXmlToDataFramePreprocessed()

        # Act
        actual = sut._normalise_protien_names(passage=text, protiens_with_no_rel_offset=offsets)

        # Assert
        self.assertEqual(expected, actual)
