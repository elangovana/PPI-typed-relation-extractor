import collections

import numpy
from sklearn import preprocessing


UNKNOWN_WORD = '<unk>'

EOS = '<eos>'


class Parser:
    def split_text(self, text, char_based=False):
        if char_based:
            return list(text)
        else:
            return text.split()

    def normalize_text(self, text):
        return text.strip().lower()

    def make_vocab(self, dataset, max_vocab_size=20000, min_freq=2, tokens_index=0):
        counts = self.get_counts_by_token(dataset, tokens_index)

        vocab = self.get_min_dictionary()
        for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if len(vocab) >= max_vocab_size or c < min_freq:
                break
            vocab[w] = len(vocab)
        return vocab

    @staticmethod
    def get_min_dictionary():
        return {EOS: 0, UNKNOWN_WORD: 1}

    def get_counts_by_token(self, dataset, tokens_index=0):
        counts = collections.defaultdict(int)
        for record in dataset:
            tokens = record[tokens_index]
            for token in tokens:
                counts[token] += 1
        return counts

    def make_array(self, tokens, vocab, add_eos=True):
        unk_id = vocab[UNKNOWN_WORD]
        eos_id = vocab[EOS]
        ids = [vocab.get(token, unk_id) for token in tokens]
        if add_eos:
            ids.append(eos_id)
        return numpy.array(ids, numpy.int32)

    def transform_to_array(self, dataset, vocab: dict):
        """
Transforms the dataset containing text into an dataset containing an array of vocab indices.
        :param dataset: Each row is a list of tokens
        :param vocab: Vocabulary to use
        :return: Return a list of lists, where each row consist of vocab indices except for the toke index
        """


        result = []
        for r in dataset:
            tokenised = []
            for token in r:
                tokenised.append(self.make_array(token, vocab))
            result.append(tokenised)

        return result

    def encode_labels(self, y):
        le = preprocessing.LabelEncoder()
        return le.fit_transform(y)
