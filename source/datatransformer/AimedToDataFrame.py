import argparse
import glob
import itertools
import logging
import os
import re

import pandas as pd


class AimedToDataFrame:

    def __init__(self):
        self.namespaces = {}
        # can be nested
        # <prot>  <prot>  keratinocyte growth factor </prot>  receptor </prot>

        # Extract relations, including nested on
        #  <p1  pair=1 >  <p1  pair=2 >  <p1  pair=3 >  <prot> FGF - 7 </prot>  </p1>  </p1>  </p1>
        relation_start_regex_s = r'(<p(\d)\s+pair=\s*(\d)\s*>\s*)'
        relation_end_regex_s = r'(</p\3>\s*)'
        protein_regex_s = r'<prot>\s*(.*?)\s*</prot>'

        self._relation_regex_s = r'({}+\s*{}\s*{}+)'.format(relation_start_regex_s, protein_regex_s,
                                                            relation_end_regex_s)

        self._relation_start_regex = re.compile(relation_start_regex_s)
        self._relation_regex = re.compile(self._relation_regex_s)
        self._whitespace_regex = re.compile(r'\s+')

    def __call__(self, aimed_file):
        file_name = os.path.basename(aimed_file).split(".txt")[0]
        with open(aimed_file, "r") as txt_handle:
            return self.parse(txt_handle, file_name)

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def parse(self, txt_handle, doc_id):
        result_json = self._parse_to_json(doc_id, txt_handle)

        return pd.DataFrame(result_json)

    def _parse_to_json(self, doc_id, txt_handle):
        result_json = []
        doc_id = doc_id
        for line_no, line in enumerate(txt_handle):
            json_line = self._parse_line(doc_id, line, line_no)

            if json_line is not None: result_json.extend(json_line)
        return result_json

    def _parse_line(self, doc_id, line, line_no):
        # Regex

        # find matches

        relation_protein_pairs_set = self._extract_relations(line)
        protein_names_set = self._extract_proteins(line)

        cleaned_line = self._relation_regex.sub(r'\5 ', line)
        cleaned_line = self._strip_protein_tags(cleaned_line)
        cleaned_line = self._whitespace_regex.sub(" ", cleaned_line)
        cleaned_line = cleaned_line.strip("\n")

        result_json = []

        for protein_combination in itertools.combinations(sorted(protein_names_set), 2):
            # sort names so it is easier to test
            protein_combination = sorted(protein_combination)
            participant1 = protein_combination[0]
            participant2 = protein_combination[1]

            protein_combination = frozenset(protein_combination)
            is_valid = protein_combination in relation_protein_pairs_set

            json_line = {"docid": doc_id,
                         "line_no": line_no + 1,
                         "passage": cleaned_line,
                         "participant1": participant1,
                         "participant2": participant2,
                         "isValid": is_valid}

            result_json.append(json_line)

        # Case just on protein found:
        # TODO: no self relations allowed now..

        return result_json

    def _strip_protein_tags(self, text_to_clean):
        cleaned_line = text_to_clean.replace("<prot>", "").replace("</prot>", "")
        cleaned_line = self._whitespace_regex.sub(" ", cleaned_line)
        return cleaned_line

    def load_dir(self, dir):
        assert os.path.isdir(dir), "{} must be a directory".format(dir)

        files = glob.glob("{}/*".format(dir.rstrip("/")))
        result = []
        for input_file in files:
            self._logger.info("Processing file {}".format(input_file))
            with open(input_file, "r") as f:
                result.extend(self._parse_to_json(os.path.basename(input_file), f))

        return pd.DataFrame(result)

    # def _extract_relations(self, text):
    #     result = {}
    #     relation_pair_matched = self._relation_regex.findall(text)
    #
    #     for r in relation_pair_matched:
    #         # loop through protein name
    #         protein_names = self._extract_proteins(r[0])
    #         # loop through nested rel
    #         rel_starts = self._relation_start_regex.findall(r[0])
    #
    #         for rs in rel_starts:
    #             node_type = "src" if rs[1] == "1" else "dest"
    #
    #             pair_id = rs[2]
    #
    #             for protein_name in protein_names:
    #                 # add <p1> as the key
    #                 if pair_id not in result:
    #                     result[pair_id] = {}
    #                 if node_type not in result[pair_id]:
    #                     result[pair_id][node_type] = []
    #                 result[pair_id][node_type].append(protein_name)
    #
    #     protein_pairs = []
    #     for _, pair in result.items():
    #         for src in sorted(pair.get("src",[])):
    #             for dest in sorted(pair.get("dest",[])):
    #                 protein_pairs.append(frozenset(sorted([src, dest])))
    #
    #     return protein_pairs

    def _extract_relations(self, text):
        src_proteins = self._parse_start_rel(text, "<p1", "</p1>")

        dest_proteins = self._parse_start_rel(text, "<p2", "</p2>")

        protein_pairs = []
        for pair_id in sorted(src_proteins.keys()):
            for src in sorted(src_proteins[pair_id]):
                for dest in sorted(dest_proteins.get(pair_id, [])):
                    protein_pairs.append(frozenset(sorted([src, dest])))

        return protein_pairs

    def _parse_start_rel(self, text, start_tag, end_tag):
        result = {}
        src_stack = []

        words = text.split(" ")
        for i, w in enumerate(words):
            if w.startswith(start_tag):
                j = i
                while (not words[j].startswith("pair=")):
                    j += 1

                src_stack.append((j , words[j].strip("pair=")))

            if w == end_tag:
                w_start_index, pair_id = src_stack.pop()
                proteins = self._extract_proteins(" ".join(words[w_start_index:i]))
                result[pair_id] = proteins

        return result

    def _extract_proteins(self, text):
        words = text.split(" ")
        stack = []
        result = set()
        for i, w in enumerate(words):
            if w == '<prot>':
                stack.append(i + 1)

            if w == '</prot>':
                w_start_index = stack.pop()
                protein_name = " ".join(words[w_start_index:i])
                protein_name = self._strip_protein_tags(protein_name).strip(" ")
                result = result.union([protein_name])

        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir",
                        help="The dir file containing abstracts")

    parser.add_argument("output",
                        help="The output_file")

    args = parser.parse_args()

    # Run
    result = AimedToDataFrame().load_dir(args.input_dir)
    result.to_json(args.output)
