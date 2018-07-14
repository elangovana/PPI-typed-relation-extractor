import lxml.etree as ET
import os


class DataPreprocessor:

    def transform(self, xmlHandle, outputHandle):
        dom = ET.parse(xmlHandle)
        fulXsltFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flatten.xslt")
        with (open(fulXsltFilePath, "rb")) as xsltHandle:
            xslt = ET.parse(xsltHandle)
            transform = ET.XSLT(xslt)
            newdom = transform(dom)

        newdom.write(outputHandle)
        return outputHandle
