import logging

import torch
import torch.nn as nn


class RelationExtractorBiLstmNetworkNoPos(nn.Module):
    """
    Implement paper:
Identifying Protein-protein Interactions in Biomedical Literature using Recurrent Neural Networks with Long Short-Term Memory. / Hsieh, Yu Lun; Chang, Yung-Chun; Chang, Nai Wen; Hsu, Wen Lian.

Proceedings of the Eighth International Joint Conference on Natural Language Processing. Asian Federation of Natural Language Processing, 2017. p. 240-245 (Proceedings of the Eighth International Joint Conference on Natural Language Processing).
    """

    def __init__(self, class_size, embedding_dim, feature_lengths, embed_vocab_size=0, seed=None,
                 hidden_size=75, dropout_rate_fc=0.2, num_layers=2, lstm_dropout=.3):
        self.embed_vocab_size = embed_vocab_size
        self.feature_lengths = feature_lengths
        if seed is None:
            seed = torch.initial_seed() & ((1 << 63) - 1)
        self.logger.info("Using seed {}".format(seed))
        torch.manual_seed(seed)

        super(RelationExtractorBiLstmNetworkNoPos, self).__init__()
        # Use random weights if vocab size if passed in else load pretrained weights

        self.set_embeddings(None)
        self.embedding_dim = embedding_dim
        self.text_column_index = self.feature_lengths.argmax(axis=0)

        self.max_sequence_len = self.feature_lengths[self.text_column_index]

        self.text_column_index = self.feature_lengths.argmax(axis=0)

        self.logger.info("The text feature is index {}, the feature lengths are {}".format(self.text_column_index,
                                                                                           self.feature_lengths))

        total_dim_size = embedding_dim

        bidirectional = True
        num_directions = 2 if bidirectional else 1

        self.lstm = nn.Sequential(
            nn.Dropout(p=lstm_dropout),
            nn.LSTM(total_dim_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                    bidirectional=bidirectional))

        #
        self.fc_input_size = (hidden_size * num_directions) * self.max_sequence_len

        self._class_size = class_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate_fc),
            nn.Linear(self.fc_input_size,
                      class_size))
        # No softmax
        # nn.LogSoftmax())

    @property
    def embeddings(self):
        if self.__embeddings is None:
            assert self.embed_vocab_size > 0, "Please set the vocab size for using random embeddings "
            self.__embeddings = nn.Embedding(self.embed_vocab_size, self.embedding_dim)
            self.__embeddings.weight.requires_grad = True

        return self.__embeddings

    def set_embeddings(self, value):
        self.__embeddings = value
        if self.__embeddings is not None:
            self.__embeddings.weight.requires_grad = True

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def forward(self, feature_tuples):

        # The input format is tuples of features.. where each item in tuple is a shape feature_len * batch_szie

        # Assume text is when the feature length is max..

        text_inputs = feature_tuples[self.text_column_index]

        text_transposed = text_inputs

        self.logger.debug("Executing embeddings")
        embeddings = self.embeddings(text_transposed)

        self.logger.debug("Running through layers")
        outputs, (_, _) = self.lstm(embeddings)

        # transform such that the shape is batch, seq, embedding
        outputs = outputs.permute(0, 2, 1).contiguous()

        # out = outputs[:, last_time_step_index, :]

        self.logger.debug("Running fc")
        # out = self.fc(out)

        out = outputs.view(-1, self.fc_input_size)
        out = self.fc(out)
        # self.logger.debug("Running softmax")
        # log_probs = self.softmax(out, dim=1)
        return out
