import logging

import torch
from pytorch_pretrained_bert import BertTokenizer


class TransformBertTextTokenToIndex:
    """
    Transform token to index
    """

    def __init__(self, bert_model_dir, case_insensitive=False):
        self.case_insensitive = case_insensitive

        self.tokeniser = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=case_insensitive)

        # Load pretrained vocab

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def construct_vocab_dict(self, data_loader):
        return self.tokeniser.vocab

    @property
    def vocab_dict(self):
        return self.tokeniser.vocab

    @vocab_dict.setter
    def vocab_dict(self, vocab_index):
        self.logger.warning("Vocab will not be updated...")

    def fit(self, data_loader):
        # Do Nothing
        pass

    def transform(self, x):
        self.logger.info("Transforming TransformBertTextTokenToIndex")
        id_converter = self.tokeniser.convert_tokens_to_ids

        batches = []
        for idx, b in enumerate(x):
            b_x = b[0]
            b_y = b[1]
            col = []
            for c_index, c in enumerate(b_x):
                row = []
                for _, r in enumerate(c):
                    tokens = id_converter(r)
                    row.append(tokens)
                row = torch.Tensor(row).long()
                col.append(row)

            batches.append([col, b_y])
        self.logger.info("Completed TransformBertTextTokenToIndex")
        return batches

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)
