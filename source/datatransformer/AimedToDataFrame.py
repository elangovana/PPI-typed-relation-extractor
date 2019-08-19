import argparse
import glob
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

            if json_line is not None: result_json.append(json_line)
        return result_json

    def _parse_line(self, doc_id, line, line_no):
        # Regex
        protein_regex_str = r'<prot>\s*(.*?)\s*</prot>'
        protein_regex = re.compile(protein_regex_str)
        whitespace_regex = re.compile(r"\s+")
        relation_regex = re.compile(r'(<p(\d)\s+pair=\d\s*>\s*({})\s*</p\2>)'.format(protein_regex_str))

        # find matches
        protien_matches = protein_regex.findall(line)
        relation_pair_matched = relation_regex.findall(line)
        assert len(
            protien_matches) <= 2, "Maximum match of proteins per sentcence is 2, but found {} in line {}".format(
            len(protien_matches), line)
        assert len(
            relation_pair_matched) in [0,
                                       2], \
            "Maximum match of relation_pair_matched per sentence is either zero or 1, but found {} in line {}".format(
                len(relation_pair_matched), line)

        # No proteins , so return none
        if len(protien_matches) == 0:
            return None

        # p1
        p1 = protien_matches[0]

        p2 = None
        if len(protien_matches) == 2:
            p2 = protien_matches[1]

        cleaned_line = relation_regex.sub(r'\3', line)
        cleaned_line = protein_regex.sub(r'\1', cleaned_line)
        cleaned_line = whitespace_regex.sub(" ", cleaned_line)
        cleaned_line = cleaned_line.strip("\n")

        is_valid = len(relation_pair_matched) > 0

        json_line = {"docid": doc_id,
                     "line_no": line_no + 1,
                     "passage": cleaned_line,
                     "participant1": p1,
                     "participant2": p2,
                     "isValid": is_valid}

        return json_line

    def load_dir(self, dir):
        assert os.path.isdir(dir), "{} must be a directory".format(dir)

        files = glob.glob("{}/*".format(dir.rstrip("/")))
        result = []
        for input_file in files:
            self._logger.info("Processing file {}".format(input_file))
            with open(input_file, "r") as f:
                result.extend(self._parse_to_json(os.path.basename(input_file), f))

        return pd.DataFrame(result)


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
