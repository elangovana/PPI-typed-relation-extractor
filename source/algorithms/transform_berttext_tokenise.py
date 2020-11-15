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

    @staticmethod
    def unk_token():
        return "[UNK]"

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
        unknown_tokens = 0
        for idx, b in enumerate(x):
            b_x = b[0]
            b_y = b[1]
            col = []
            for c_index, c in enumerate(b_x):
                row = []
                max = self.max_feature_lens[c_index]
                for _, r in enumerate(c):
                    all_tokens = tokeniser(r)
                    sized_tokens = all_tokens[0:max - 2]
                    unknown_tokens += sum([1 for t in sized_tokens if t == self.unk_token()])
                    sized_tokens = ['[CLS]'] + sized_tokens + [pad] * (max - 2 - len(sized_tokens)) + ['[SEP]']
                    row.append(sized_tokens)
                col.append(row)

            batches.append([col, b_y])
        self.logger.info("Unknown tokens count {}".format(unknown_tokens))

        self.logger.info("Completed TransformBertTextTokenise")
        return batches

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)
