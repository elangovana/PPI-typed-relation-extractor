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
        protein_regex_str = r'<prot>\s*(.*?)\s*</prot>'
        protein_regex = re.compile(protein_regex_str)
        whitespace_regex = re.compile(r"\s+")
        relation_regex = re.compile(r'(<p(\d)\s+pair=(\d)\s*>\s*({})\s*</p\2>)'.format(protein_regex_str))

        # find matches
        protien_matches = protein_regex.findall(line)
        relation_pair_matched = relation_regex.findall(line)

        relation_protein_pairs_set = self._extract_relations(relation_pair_matched)
        protein_names_set = self._extract_proteins(protien_matches)

        cleaned_line = relation_regex.sub(r'\4', line)
        cleaned_line = protein_regex.sub(r'\1', cleaned_line)
        cleaned_line = whitespace_regex.sub(" ", cleaned_line)
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

    def load_dir(self, dir):
        assert os.path.isdir(dir), "{} must be a directory".format(dir)

        files = glob.glob("{}/*".format(dir.rstrip("/")))
        result = []
        for input_file in files:
            self._logger.info("Processing file {}".format(input_file))
            with open(input_file, "r") as f:
                result.extend(self._parse_to_json(os.path.basename(input_file), f))

        return pd.DataFrame(result)

    def _extract_relations(self, relation_pair_matched):
        result = {}
        for r in relation_pair_matched:
            pair_id = r[2]
            protein_name = r[4]

            if pair_id not in result:
                result[pair_id] = []

            result[pair_id].append(protein_name)

        result = set([frozenset(v) for k, v in result.items()])

        return result

    def _extract_proteins(self, proteins_matched):
        result = []
        for r in proteins_matched:
            protein_name = r
            result.append(protein_name)

        result = set(result)

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
