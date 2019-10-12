import logging

import torch
from pytorch_pretrained_bert import BertTokenizer


class TransformBertTextTokenToIndex:
    """
    Transform token to index
    """

    def __init__(self, bert_model_dir, case_insensitive=False, text_col_index=0):
        self.text_col_index = text_col_index
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
        raise NotImplementedError

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

            text_col = b_x[self.text_col_index]
            rows = []
            for _, r in enumerate(text_col):
                tokens = id_converter(r)
                rows.append(tokens)
            col = torch.Tensor(rows).long()

            batches.append([col, b_y])

        self.logger.info("Completed TransformBertTextTokenToIndex")
        return batches

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)
