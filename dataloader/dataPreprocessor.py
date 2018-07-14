import lxml.etree as ET
import os
from io import BytesIO
class DataPreprocessor:



    def transform(self, xmlHandle):
        dom = ET.parse(xmlHandle)
        fulXsltFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flatten.xslt")
        with (open(fulXsltFilePath, "rb")) as xsltHandle:
            xslt = ET.parse(xsltHandle)
            transform = ET.XSLT(xslt)
            newdom = transform(dom)

            outputHandle = BytesIO()
        newdom.write(outputHandle)
        outputHandle.seek(0)
        #yield return each data
        for entry in self._iter_elements_by_name(outputHandle, "data", {}):
            yield entry

    def adddata(self, entry):
        pubmed = ET.SubElement(entry, "pubmed")




    def _iter_elements_by_name(self, handle, name, namespace):
        events = ET.iterparse(handle, events=("start", "end"))
        _, root = next(events)  # Grab the root element.

        expanded_name = name
        # If name has the namespace, expand it
        if ":" in name :
            local_name = name[name.index(":") + 1:]
            namespace_short_name = name[:name.index(":")]
            expanded_name = "{{{}}}{}".format(namespace[namespace_short_name], local_name)

        for event, elem in events:
            if event == "end" and elem.tag == expanded_name:
                yield elem
                elem.clear()
