import logging

from pytorch_pretrained_bert import BertTokenizer


class TransformBertTextTokenise:
    """
    Extracts vocab from data frame columns which have already been tokenised into words
    """

    def __init__(self, bert_model_dir, max_feature_lens, case_insensitive=False):
        self.max_feature_lens = max_feature_lens
        self.case_insensitive = case_insensitive

        self.tokeniser = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=case_insensitive)

        # Load pretrained vocab

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def construct_vocab_dict(self, data_loader):
        return self.tokeniser.vocab

    @staticmethod
    def pad_token():
        return "[PAD]"

    @staticmethod
    def eos_token():
        return "<EOS>"

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
        self.logger.info("Transforming TransformBertTextTokenise")
        pad = self.pad_token()
        tokeniser = self.tokeniser.tokenize

        batches = []
        for idx, b in enumerate(x):
            b_x = b[0]
            b_y = b[1]
            col = []
            for c_index, c in enumerate(b_x):
                row = []
                max = self.max_feature_lens[c_index]
                for _, r in enumerate(c):
                    tokens = tokeniser(r)[0:max]
                    tokens = tokens + [pad] * (max - len(tokens))
                    row.append(tokens)
                col.append(row)

            batches.append([col, b_y])
        self.logger.info("Completed TransformBertTextTokenise")
        return batches

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)
