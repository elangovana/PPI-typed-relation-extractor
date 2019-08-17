import argparse
import itertools
from xml.etree import ElementTree

import pandas as pd


class BiocToDataFrame:

    def __init__(self):
        self.namespaces = {}

    def __call__(self, xmlbuffer_or_path):
        if isinstance(xmlbuffer_or_path, str):
            with open(xmlbuffer_or_path, "r") as xmlhandle:
                return self.parse(xmlhandle)

        return self.parse(xmlbuffer_or_path)

    def parse(self, xmlbuffer_or_path):
        result_json = []
        for document in self._iter_elements_by_name(xmlbuffer_or_path, "document", self.namespaces):
            doc_id = document.find("id").text
            passage_ele = document.find("passage")
            passage = passage_ele.find("text").text

            # Get all proteins
            proteins = [p for p in self._find_protein_annotations(passage_ele)]

            # make them unique
            proteins = set(proteins)

            rel_protein_pairs = set()
            for p1, p2 in self._find_protein_relations(passage_ele):
                rel_protein_pairs.add(frozenset([p1, p2]))

            for protein_combination in itertools.combinations(proteins, 2):
                # sort names so it is easier to test
                protein_combination = sorted(protein_combination)
                participant1 = protein_combination[0]
                participant2 = protein_combination[1]

                protein_combination = frozenset(protein_combination)
                is_valid = protein_combination in rel_protein_pairs

                result_json.append({"docid": doc_id
                                       , "passage": passage
                                       , "participant1": participant1
                                       , "participant2": participant2
                                       , "isValid": is_valid

                                    })

        return pd.DataFrame(result_json)

    @staticmethod
    def _find_protein_annotations(passage_ele):
        for annotation_ele in passage_ele.findall("annotation"):
            is_protein = False
            for infon_ele in annotation_ele.findall("infon"):
                if infon_ele.attrib["key"] == 'type' and infon_ele.text == 'protein':
                    is_protein = True
                    break
            if is_protein:
                yield annotation_ele.find("text").text

    @staticmethod
    def _find_protein_relations(passage_ele):
        for annotation_ele in passage_ele.findall("relation"):
            is_relation = False
            for infon_ele in annotation_ele.findall("infon"):
                if infon_ele.attrib["key"] == 'type' and infon_ele.text == 'Relation':
                    is_relation = True
                    break
            if is_relation:
                participant1_id = annotation_ele.find("node[@role='Arg1']").attrib["refid"]
                participant1 = passage_ele.find("annotation[@id='{}']/text".format(participant1_id)).text

                participant2_id = annotation_ele.find("node[@role='Arg2']").attrib["refid"]
                participant2 = passage_ele.find("annotation[@id='{}']/text".format(participant2_id)).text

                yield participant1, participant2

    @staticmethod
    def _iter_elements_by_name(handle, name, namespace):
        events = ElementTree.iterparse(handle, events=("start", "end"))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input",
                        help="The bioc xml formatted json")

    parser.add_argument("output",
                        help="The output_file")

    args = parser.parse_args()

    # Run
    result = BiocToDataFrame().parse(args.input)
    result.to_json(args.output)
