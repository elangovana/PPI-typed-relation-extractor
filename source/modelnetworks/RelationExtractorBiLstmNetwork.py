import logging

import torch
import torch.nn as nn

from algorithms.PositionEmbedder import PositionEmbedder


class RelationExtractorBiLstmNetwork(nn.Module):

    def __init__(self, class_size, embedding_dim, feature_lengths, embed_vocab_size=0, seed=None, pos_embedder=None,
                 hidden_size=75, dropout_rate_fc=0.2, kernal_size=4, fc_layer_size=30,
                 num_layers=2,
                 lstm_dropout=.3):
        self.embed_vocab_size = embed_vocab_size
        self.feature_lengths = feature_lengths
        if seed is None:
            seed = torch.initial_seed() & ((1 << 63) - 1)
        self.logger.info("Using seed {}".format(seed))
        torch.manual_seed(seed)

        super(RelationExtractorBiLstmNetwork, self).__init__()
        # Use random weights if vocab size if passed in else load pretrained weights

        self.set_embeddings(None)
        self.embedding_dim = embedding_dim

        self.__pos_embedder__ = pos_embedder

        self.text_column_index = self.feature_lengths.argmax(axis=0)

        self.max_sequence_len = self.feature_lengths[self.text_column_index]

        self.logger.info("The text feature is index {}, the feature lengths are {}".format(self.text_column_index,
                                                                                           self.feature_lengths))

        # The total embedding size if the text column + position for the rest
        pos_embed_total_dim = (len(self.feature_lengths) - 1) * \
                              self.pos_embedder.embeddings.shape[1]
        total_dim_size = embedding_dim + pos_embed_total_dim

        bidirectional = True
        num_directions = 2 if bidirectional else 1

        self.logger.info(
            "Word embedding size is {}, pos embedding size is {}, totaldim is {}, hidden_size  {}".format(embedding_dim,
                                                                                                          pos_embed_total_dim,
                                                                                                          total_dim_size,
                                                                                                          hidden_size
                                                                                                          ))

        self.lstm = nn.Sequential(
            nn.LSTM(total_dim_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                    bidirectional=bidirectional, dropout=lstm_dropout))

        self.max_pooling = nn.MaxPool1d(kernel_size=kernal_size)
        self.avg_pooling = nn.AvgPool1d(kernel_size=kernal_size)
        #
        self.fc_input_size = (self.max_sequence_len // kernal_size) * 2 * (hidden_size * num_directions)

        self._class_size = class_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate_fc),
            nn.Linear(self.fc_input_size,
                      fc_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate_fc),
            nn.Linear(fc_layer_size, class_size))
        # No softmax
        # nn.LogSoftmax())

    @property
    def embeddings(self):
        if self.__embeddings is None:
            assert self.embed_vocab_size > 0, "Please set the vocab size for using random embeddings "
            self.__embeddings = nn.Embedding(self.embed_vocab_size, self.embedding_dim)
        return self.__embeddings

    def set_embeddings(self, value):
        self.__embeddings = value

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def pos_embedder(self):
        self.__pos_embedder__ = self.__pos_embedder__ or PositionEmbedder()
        return self.__pos_embedder__

    def forward(self, feature_tuples):

        # The input format is tuples of features.. where each item in tuple is a shape feature_len * batch_szie

        # Assume text is when the feature length is max..

        text_inputs = feature_tuples[self.text_column_index]

        text_transposed = text_inputs

        self.logger.debug("Executing embeddings")
        embeddings = self.embeddings(text_transposed)

        # TODO: avoid this loop, use builtin
        embeddings_with_pos = embeddings
        self.logger.debug("Executing pos embedding")

        for f in range(len(feature_tuples)):
            if f == self.text_column_index: continue

            entity = feature_tuples[f]  # .transpose(0, 1)

            # TODO: avoid this loop, use builtin
            batch_pos_embedding_entity = []

            for i, (t, e, sentence_embedding) in enumerate(zip(text_transposed, entity, embeddings)):
                sentence_pos_embedding = self.pos_embedder(t, e[0])

                # Set pos_embedding to zero when pad token ( indicated by zero embedding)
                sentence_pos_embedding[torch.all(sentence_embedding.eq(0.0), dim=1)] = 0.0
                batch_pos_embedding_entity.append(sentence_pos_embedding)

            batch_pos_embedding_entity_tensor = torch.stack(batch_pos_embedding_entity).to(
                device=embeddings_with_pos.device)

            embeddings_with_pos = torch.cat([embeddings_with_pos, batch_pos_embedding_entity_tensor], dim=2)
        # Final output
        # reshape takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        # final_input = merged_pos_embed.permute(0, 2, 1)

        self.logger.debug("Running through layers")
        outputs, (_, _) = self.lstm(embeddings_with_pos)

        outputs = outputs.permute(0, 2, 1)

        max_pool_out = self.max_pooling(outputs)

        avg_pool_out = self.avg_pooling(outputs)

        # out = outputs[:, last_time_step_index, :]

        self.logger.debug("Running fc")
        # out = self.fc(out)
        out = torch.cat([max_pool_out, avg_pool_out], dim=2)

        out = out.view(-1, self.fc_input_size)
        out = self.fc(out)
        # self.logger.debug("Running softmax")
        # log_probs = self.softmax(out, dim=1)
        return out
