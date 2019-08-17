import os
from io import StringIO
from unittest import TestCase

import pandas as pd

from datatransformer.BiocToDataFrame import BiocToDataFrame


class TestBiocToDataFrameJson(TestCase):
    def test___call__no_relation(self):
        # Arrange
        xml = """<?xml version="1.0" encoding="UTF-8"?>
    <collection>
    <source></source>
    <date></date>
    <key></key>
    <document>
        <id>AIMed_d30</id>
        <passage>
            <offset>0</offset>
            <text>Simple text 1</text>
            <annotation id="T1">
                <infon key="file">ann</infon>
                <infon key="type">protein</infon>
                <location offset="19" length="13"></location>
                <text>delta-catenin</text>
            </annotation>
            <annotation id="T3">
                <infon key="file">ann</infon>
                <infon key="type">protein</infon>
                <location offset="122" length="17"></location>
                <text>presenilin (PS) 1</text>
            </annotation>
          <annotation id="T2">
                <infon key="file">ann</infon>
                <infon key="type">protein</infon>
                <location offset="122" length="17"></location>
                <text>presenilin (PS) 1</text>
            </annotation>
        </passage>
    </document>
    
    </collection>
        """

        xml_handle = StringIO(xml)

        sut = BiocToDataFrame()

        expected_json = [{"docid": "AIMed_d30",
                          "passage": "Simple text 1"
                             , "participant1": "delta-catenin"
                             , "participant2": "presenilin (PS) 1"
                             , "isValid": False

                          }]

        expected_df = pd.DataFrame(expected_json)

        # Act
        dataframe = sut(xml_handle)

        # Assert
        self.assertSequenceEqual(dataframe.values.tolist(), expected_df.values.tolist())

    def test___call___yes_relation(self):
        # Arrange
        xml = """<?xml version="1.0" encoding="UTF-8"?>
    <collection>
    <source></source>
    <date></date>
    <key></key>
    <document>
        <id>AIMed_d30</id>
        <passage>
            <offset>0</offset>
            <text>Simple text 1</text>
            <annotation id="T1">
                <infon key="file">ann</infon>
                <infon key="type">protein</infon>
                <location offset="19" length="13"></location>
                <text>delta-catenin</text>
            </annotation>
            <annotation id="T2">
                <infon key="file">ann</infon>
                <infon key="type">protein</infon>
                <location offset="122" length="17"></location>
                <text>presenilin (PS) 1</text>
            </annotation>
            <relation id="R1">
                            <infon key="relation type">Interaction</infon>
                            <infon key="file">ann</infon>
                            <infon key="type">Relation</infon>
                            <node refid="T1" role="Arg1"></node>
                            <node refid="T2" role="Arg2"></node>
                </relation>
        </passage>
    </document>

    </collection>
        """

        xml_handle = StringIO(xml)

        sut = BiocToDataFrame()

        expected_json = [{"docid": "AIMed_d30"
                             , "passage": "Simple text 1"
                             , "participant1": "delta-catenin"
                             , "participant2": "presenilin (PS) 1"
                             , "isValid": True

                          }]

        expected_df = pd.DataFrame(expected_json)

        # Act
        dataframe = sut(xml_handle)

        # Assert
        self.assertSequenceEqual(dataframe.values.tolist(), expected_df.values.tolist())

    def test___call__file(self):
        # Arrange
        file = os.path.join(os.path.dirname(__file__), "data", "sample_aimed.xml")

        sut = BiocToDataFrame()

        # Act
        dataframe = sut(file)

        # Assert
        self.assertIsInstance(dataframe, pd.DataFrame)
