import argparse
import itertools
import re
from xml.etree import ElementTree

import pandas as pd


class AimedXmlToDataFramePreprocessed:

    def __init__(self):

        self.namespaces = {}

    def __call__(self, xmlbuffer_or_path):
        """
        The xml_File must be formatted as per http://mars.cs.utu.fi/PPICorpora/
         1. Download from ftp://ftp.cs.utexas.edu/pub/mooney/bio-data/interactions.tar.gz"
         2. Convert the raw dataset into XML for using instructions in http://mars.cs.utu.fi/PPICorpora/
            convert_aimed.py -i  aimed_interactions_input_dir -o aimed.xml

        Acknowledgements: Pyysalo S, Airola A, Heimonen J, Björne J, Ginter F, Salakoski T, Comparative Analysis of Five Protein-protein Interaction Corpora, LBM'07. 2007.
         """
        if isinstance(xmlbuffer_or_path, str):
            with open(xmlbuffer_or_path, "r") as xmlhandle:
                return self.parse(xmlhandle)

        return self.parse(xmlbuffer_or_path)

    def parse(self, xmlbuffer_or_path):
        result_json = []
        for document in self._iter_elements_by_name(xmlbuffer_or_path, "document", self.namespaces):
            doc_id = document.attrib["id"]

            for passage_ele in document.findall("sentence"):
                passage = passage_ele.attrib["text"]
                passageid = passage_ele.attrib["id"]

                # Get all proteins and offsets in the sentence
                protein_offsets_dict = {l: p for p, l in self._find_protein_annotations(passage_ele)}

                rel_protein_pairs = set()
                for (p1, p1_l), (p2, p2_l) in self._find_protein_relations(passage_ele):
                    pair = frozenset([p1_l, p2_l])
                    rel_protein_pairs.add(pair)

                comb_func = itertools.combinations

                for protein_combination in comb_func(set(protein_offsets_dict.keys()), 2, ):
                    # sort names so it is easier to test
                    protein_combination = sorted(protein_combination)
                    participant1_offset = protein_combination[0]
                    participant1 = protein_offsets_dict[participant1_offset]

                    participant2_offset = protein_combination[1]
                    participant2 = protein_offsets_dict[participant2_offset]

                    protein_combination = frozenset(protein_combination)
                    is_valid = protein_combination in rel_protein_pairs

                    protiens_with_no_rel_offset = set(protein_offsets_dict.keys()) - protein_combination

                    normalised_passage = self._normalise_protien_names(passage, protiens_with_no_rel_offset)

                    result_json.append({"docid": doc_id
                                           , "passage": normalised_passage
                                           , "passageid": passageid
                                           , "participant1": participant1
                                           , "participant1_loc": participant1_offset

                                           , "participant2": participant2
                                           , "participant2_loc": participant2_offset

                                           , "isValid": is_valid

                                        })

        return pd.DataFrame(result_json)

    @staticmethod
    def _find_protein_annotations(passage_ele):
        for entity_ele in passage_ele.findall("entity"):
            if entity_ele.attrib["type"] != 'protein': continue

            yield (entity_ele.attrib["text"], entity_ele.attrib["charOffset"])

    @staticmethod
    def _find_protein_relations(passage_ele):
        """
    Finds the interations from a sentenced block
    <sentence id="AIMed.d28.s234" text="We have identified a new TNF-related ligand, designated human GITR ligand (hGITRL), and its human receptor (hGITR), an ortholog of the recently discovered murine glucocorticoid-induced TNFR-related (mGITR) protein [4]." seqId="s234">
      <entity id="AIMed.d28.s234.e0" charOffset="62-65" type="protein" text="GITR" seqId="e329"/>
      <entity id="AIMed.d28.s234.e1" charOffset="62-72" type="protein" text="GITR ligand" seqId="e330"/>
      <entity id="AIMed.d28.s234.e2" charOffset="75-80" type="protein" text="hGITRL" seqId="e332"/>
      <entity id="AIMed.d28.s234.e3" charOffset="108-112" type="protein" text="hGITR" seqId="e335"/>
      <entity id="AIMed.d28.s234.e4" charOffset="199-203" type="protein" text="mGITR" seqId="e338"/>
      <entity id="AIMed.d28.s234.e5" charOffset="162-212" type="protein" text="glucocorticoid-induced TNFR-related (mGITR) protein" seqId="e339"/>
      <interaction id="AIMed.d28.s234.i0" e1="AIMed.d28.s234.e0" e2="AIMed.d28.s234.e2" type="None" directed="false" seqId="i0"/>
      <interaction id="AIMed.d28.s234.i1" e1="AIMed.d28.s234.e1" e2="AIMed.d28.s234.e3" type="None" directed="false" seqId="i1"/>
      <interaction id="AIMed.d28.s234.i2" e1="AIMed.d28.s234.e2" e2="AIMed.d28.s234.e3" type="None" directed="false" seqId="i2"/>
    </sentence>
        :param passage_ele:
        :return:
        """
        for interaction_ele in passage_ele.findall("interaction"):
            participant1_id = interaction_ele.attrib["e1"]
            participant1_entity_ele = passage_ele.find("entity[@id='{}']".format(participant1_id))
            participant1 = participant1_entity_ele.attrib["text"]
            participant1_offset = participant1_entity_ele.attrib["charOffset"]

            participant2_id = interaction_ele.attrib["e2"]
            participant2_entity_ele = passage_ele.find("entity[@id='{}']".format(participant2_id))
            participant2 = participant2_entity_ele.attrib["text"]
            participant2_offset = participant2_entity_ele.attrib["charOffset"]

            yield (participant1, participant1_offset), (participant2, participant2_offset)

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

    def _normalise_protien_names(self, passage, protiens_with_no_rel_offset):
        offset_dict = {}
        for i in protiens_with_no_rel_offset:
            start_pos, end_pos = int(i.split("-")[0]), int(i.split("-")[1])
            offset_dict[start_pos] = end_pos

        adjustment = 0
        for s, e in sorted(offset_dict.items(), key=lambda kv: kv[0], reverse=False):
            mask = " PROTEIN "
            adj_s = s + adjustment
            adj_e = e + adjustment

            passage = passage[:adj_s] + mask + passage[adj_e + 1:]
            adjustment += len(mask) - (adj_e - adj_s) - 1

        clean_passage = re.sub("\s+", " ", passage)
        return clean_passage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input",
                        help="The input xml file formatted as instructed in http://mars.cs.utu.fi/PPICorpora/, pyysalo et.al")

    parser.add_argument("output",
                        help="The output_file")

    args = parser.parse_args()

    # Run
    result = AimedXmlToDataFramePreprocessed()(args.input)
    result.to_json(args.output)
