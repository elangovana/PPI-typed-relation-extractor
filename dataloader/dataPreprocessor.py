import json
import logging

import lxml.etree as ET
import os
from io import BytesIO, StringIO

import xmltodict

from dataloader.PubmedAbstractExtractor import PubmedAbstractExtractor


class DataPreprocessor:

    def __init__(self, pubmed_extractor=None):
        self.pubmed_extractor = pubmed_extractor
        self._logger = logging.getLogger(__name__)

    @property
    def pubmed_extractor(self):
        self.__pubmed_extractor = self.__pubmed_extractor or PubmedAbstractExtractor()
        return self.__pubmed_extractor

    @pubmed_extractor.setter
    def pubmed_extractor(self, extractor):
        self.__pubmed_extractor = extractor

    def transform(self, xmlHandle):
        """
        Transforms the xml document and iteratively returns a string of the data
        :rtype: str
        :param xmlHandle:
        """
        dom = ET.parse(xmlHandle)
        fulXsltFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flatten.xslt")

        # Transform
        with (open(fulXsltFilePath, "rb")) as xsltHandle:
            xslt = ET.parse(xsltHandle)
            transform = ET.XSLT(xslt)
            newdom = transform(dom)
            outputHandle = BytesIO()
            newdom.write(outputHandle)

        # yield return each data
        outputHandle.seek(0)
        for entry in self._iter_elements_by_name(outputHandle, "data", {}):
            yield ET.tostring(entry).decode()

    def adddata(self, entry_handle):
        # create an entry containing pubmed abstract
        tree = ET.parse(entry_handle)
        data_ele = tree.getroot()
        pubmed_id = data_ele.find("pubmedid").text

        abstract = self.pubmed_extractor.extract_abstract_by_pubmedid([pubmed_id])[0]

        # add abstract to element
        abstract_ele = ET.SubElement(data_ele, "abstract")
        abstract_ele.text = abstract

        return ET.tostring(data_ele).decode()

    def run_pipeline(self, xmlHandle):
        """
Runs the preprocessing pipeline
        :param xmlHandle: The xml stream containing data in PSI format

        """
        for data in self.transform(xmlHandle):
            yield self.convert_to_json(StringIO(self.adddata(data)))


    def _iter_elements_by_name(self, handle, name, namespace):
        events = ET.iterparse(handle, events=("start", "end"))
        _, root = next(events)  # Grab the root element.

        expanded_name = name
        # If name has the namespace, expand it
        if ":" in name:
            local_name = name[name.index(":") + 1:]
            namespace_short_name = name[:name.index(":")]
            expanded_name = "{{{}}}{}".format(namespace[namespace_short_name], local_name)

        for event, elem in events:
            if event == "end" and elem.tag == expanded_name:
                yield elem
                elem.clear()

    def convert_to_json(self, xmlHandle):
        return json.dumps(xmltodict.parse(xmlHandle))
