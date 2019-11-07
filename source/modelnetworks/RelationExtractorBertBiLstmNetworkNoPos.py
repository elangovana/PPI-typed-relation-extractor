import logging

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertForSequenceClassification


class RelationExtractorBertBiLstmNetworkNoPos(nn.Module):
    """
    Implement simple BiLstm but use bert tokensier and embedding
    """

    def __init__(self, bert_model_dir, class_size, feature_lens, seed=None,
                 hidden_size=75, dropout_rate_fc=0.2, num_layers=2, input_dropout=.3, lstm_dropout=0.5,
                 fine_tune_embeddings=True):
        self.fine_tune_embeddings = fine_tune_embeddings
        if seed is None:
            seed = torch.initial_seed() & ((1 << 63) - 1)
        self.logger.info("Using seed {}".format(seed))
        torch.manual_seed(seed)

        super(RelationExtractorBertBiLstmNetworkNoPos, self).__init__()
        # Use random weights if vocab size if passed in else load pretrained weights

        self.text_column_index = feature_lens.argmax(axis=0)

        self.max_sequence_len = feature_lens[self.text_column_index]

        self.logger.info(
            "The text feature is index {}, the feature lengths are {}".format(self.text_column_index, feature_lens))

        bidirectional = True
        num_directions = 2 if bidirectional else 1

        self.embeddings = BertForSequenceClassification.from_pretrained(bert_model_dir,
                                                                        num_labels=class_size).bert.embeddings.word_embeddings

        self.lstm = nn.Sequential(
            nn.Dropout(p=input_dropout),
            nn.LSTM(self.embeddings.embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                    bidirectional=bidirectional, dropout=lstm_dropout))

        #
        self.fc_input_size = (hidden_size * num_directions) * self.max_sequence_len

        self._class_size = class_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate_fc),
            nn.Linear(self.fc_input_size,
                      class_size))

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def forward(self, input):
        # The input format is tuples of features.. where each item in tuple is a shape feature_len * batch_szie

        # Assume text is when the feature length is max..

        self.logger.debug("Executing embeddings")
        embeddings = self.embeddings(input)

        self.logger.debug("Running through layers")
        outputs, (_, _) = self.lstm(embeddings)

        # transform such that the shape is batch, seq, embedding
        outputs = outputs.permute(0, 2, 1).contiguous()

        # out = outputs[:, last_time_step_index, :]

        self.logger.debug("Running fc")
        # out = self.fc(out)

        out = outputs.view(-1, self.fc_input_size)
        out = self.fc(out)

        return out
