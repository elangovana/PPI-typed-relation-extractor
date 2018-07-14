import lxml.etree as ET

class DataPreprocessor:

    def transform(self, xmlHandle, xsltHandle,outputHandle ):
        dom = ET.parse(xmlHandle)
        xslt = ET.parse(xsltHandle)
        transform = ET.XSLT(xslt)
        newdom = transform(dom)

        newdom.write(outputHandle)
        return outputHandle
