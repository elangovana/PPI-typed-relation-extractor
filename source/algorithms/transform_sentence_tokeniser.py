import logging
import spacy

"""
Replaces the name of the protein in the abstract .
"""


class TransformSentenceTokenisor:

    def __init__(self, text_column_index, eos_token="<EOS>"):

        self.text_column_index = text_column_index
        self.eos_token = eos_token
        self.sentence_tokenisor = None

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, data_loader):
        pass

    @property
    def sentence_tokenisor(self):
        if self._sentence_tokenisor is None:
            return self._get_default_tokenisor()
        return self._sentence_tokenisor

    def _get_default_tokenisor(self):
        sentence_tokenisor = spacy.load("en_core_web_sm")
        return lambda x: [s.text.rstrip(".") for s in sentence_tokenisor(x, ).sents]

    @sentence_tokenisor.setter
    def sentence_tokenisor(self, value):
        self._sentence_tokenisor = value

    def transform(self, x):
        self.logger.info("Running sentence tokenisor ")
        eos = " {} ".format(self.eos_token)
        batches = []
        tokenisor = self.sentence_tokenisor

        for idx, (b_x, b_y) in enumerate(x):

            text_column = b_x[self.text_column_index]

            tokenised_sentences = []
            for _, t in enumerate(text_column):
                sentences = tokenisor(t)
                tokenised_sentences.append(eos.join(sentences))

            transformed_b_x = b_x
            transformed_b_x[self.text_column_index] = tokenised_sentences

            batches.append([transformed_b_x, b_y])
        self.logger.info("Completed  sentence tokenisor ")

        return batches

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)
