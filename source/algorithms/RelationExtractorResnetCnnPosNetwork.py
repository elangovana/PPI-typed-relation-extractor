import logging
import math

import torch
import torch.nn as nn

from algorithms.PositionEmbedder import PositionEmbedder


class RelationExtractorResnetCnnPosNetwork(nn.Module):

    def __init__(self, class_size, embedding_dim, feature_lengths, embed_vocab_size=0, pos_embedder=None,
                 windows_size=3, dropout_rate_cnn=.5, cnn_output=64, cnn_num_layers=3, cnn_stride=1, pool_kernel=3,
                 pool_stride=2, fc_layer_size=256, fc_dropout_rate=.5, input_dropout_rate=.2, seed=777):
        self.embed_vocab_size = embed_vocab_size
        self.feature_lengths = feature_lengths
        torch.manual_seed(seed)

        super(RelationExtractorResnetCnnPosNetwork, self).__init__()
        self.logger.info("Windows Size is {}".format(windows_size))
        # Use random weights if vocab size if passed in else load pretrained weights

        self.set_embeddings(None)
        self.embedding_dim = embedding_dim

        self.__pos_embedder__ = pos_embedder

        self.text_column_index = self.feature_lengths.argmax(axis=0)
        self.text_column_size = int(self.feature_lengths[self.text_column_index])

        self.logger.info("The text feature is index {}, the feature lengths are {}".format(self.text_column_index,
                                                                                           self.feature_lengths))

        self.windows_size = windows_size
        self.num_layers = cnn_num_layers

        total_cnn_out_size = 0
        # The total embedding size if the text column + position for the rest
        pos_embed_total_dim = (len(self.feature_lengths) - 1) * \
                              self.pos_embedder.embeddings.shape[1]
        total_dim_size = embedding_dim + pos_embed_total_dim
        self.logger.info(
            "Word embedding size is {}, pos embedding size is {}, cnn_output size {}, total is {}".format(embedding_dim,
                                                                                                          pos_embed_total_dim,
                                                                                                          cnn_output,
                                                                                                          total_dim_size))
        cnn_kernel = min(self.windows_size, self.text_column_size)
        cnn_padding = cnn_kernel // 2
        pool_padding = 0

        self._class_size = class_size
        self.activation = nn.ReLU()

        self.input_block = nn.Sequential(nn.Dropout(input_dropout_rate),
                                         nn.Conv1d(total_dim_size, cnn_output, kernel_size=cnn_kernel,
                                                   stride=cnn_stride,
                                                   padding=cnn_padding),
                                         nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride,
                                                      padding=pool_padding)
                                         )
        self.resnet_blocks = nn.ModuleList()

        cnn_out_length = math.ceil(
            (self.text_column_size + 2 * cnn_padding - cnn_kernel + 1) / cnn_stride)

        pool_out_length = math.ceil(
            (cnn_out_length + 2 * pool_padding - pool_kernel + 1) / pool_stride)

        cnn_out_length = pool_out_length
        for k in range(1, self.num_layers + 1):
            cnn_out_length = math.ceil(
                (cnn_out_length + 2 * cnn_padding - cnn_kernel + 1) / cnn_stride)
            # twice for 2 layers of CNN
            cnn_out_length = math.ceil(
                (cnn_out_length + 2 * cnn_padding - cnn_kernel + 1) / cnn_stride)

            resnetblock = nn.Sequential(
                nn.Conv1d(cnn_output, cnn_output, kernel_size=cnn_kernel, stride=cnn_stride, padding=cnn_padding),
                nn.BatchNorm1d(cnn_output),
                nn.ReLU(),
                nn.Conv1d(cnn_output, cnn_output, kernel_size=cnn_kernel, stride=cnn_stride, padding=cnn_padding),
                nn.BatchNorm1d(cnn_output)
            )

            self.resnet_blocks.add_module("ResidualBlock_{}".format(k), resnetblock)

        pool_out_length = math.ceil(
            (cnn_out_length + 2 * pool_padding - pool_kernel + 1) / pool_stride)

        self.final_block = nn.Sequential(
            nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride,
                         padding=pool_padding)
            , nn.Dropout(p=dropout_rate_cnn)

        )

        self.fc = nn.Sequential(nn.Linear(cnn_output * pool_out_length,
                                          fc_layer_size),
                                nn.Dropout(fc_dropout_rate),
                                nn.ReLU(),
                                nn.Linear(fc_layer_size, class_size))

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

        merged_pos_embed = embeddings
        self.logger.debug("Executing pos embedding")

        for f in range(len(feature_tuples)):
            if f == self.text_column_index: continue

            entity = feature_tuples[f]  # .transpose(0, 1)

            # TODO: avoid this loop, use builtin
            pos_embedding = []
            for t, e in zip(text_transposed, entity):
                pos_embedding.append(self.pos_embedder(t, e[0]))

            pos_embedding_tensor = torch.stack(pos_embedding).to(device=merged_pos_embed.device)

            merged_pos_embed = torch.cat([merged_pos_embed, pos_embedding_tensor], dim=2)

        # Final output
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        final_input = merged_pos_embed.permute(0, 2, 1)

        self.logger.debug("Running input block")
        x = self.input_block(final_input)

        self.logger.debug("Running resnet block")
        for m in self.resnet_blocks:
            x = self.activation(x + m(x))

        self.logger.debug("Running final block")
        x = self.final_block(x)

        self.logger.debug("Running fc")

        out = x.reshape(x.size(0), -1)
        out = self.fc(out)

        return out
