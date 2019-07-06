import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.PositionEmbedder import PositionEmbedder


class RelationExtractorCnnPosNetwork(nn.Module):

    def __init__(self, class_size, embedding_dim, pretrained_weights_or_embed_vocab_size, feature_lengths,
                 ngram_context_size=5, seed=777, drop_rate=.1, pos_embedder=None):
        self.feature_lengths = feature_lengths
        torch.manual_seed(seed)

        super(RelationExtractorCnnPosNetwork, self).__init__()
        self.logger.info("NGram Size is {}".format(ngram_context_size))
        self.dropout_rate = drop_rate
        # Use random weights if vocab size if passed in else load pretrained weights

        self.embeddings = nn.Embedding(pretrained_weights_or_embed_vocab_size,
                                       embedding_dim) if type(
            pretrained_weights_or_embed_vocab_size) is int else nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_weights_or_embed_vocab_size))

        self.__pos_embedder__ = pos_embedder

        self.text_column_index = self.feature_lengths.argmax(axis=0)

        self.logger.info("The text feature is index {}, the feature lengths are {}".format(self.text_column_index,
                                                                                           self.feature_lengths))

        # self.windows_sizes = [5, 4, 3, 2, 1]
        self.windows_sizes = [3, 2, 1]
        cnn_output = 10
        cnn_stride = 1
        pool_stride = 1

        self.cnn_layers = []
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

        for k in self.windows_sizes:
            layer1_cnn_output = cnn_output
            layer1_cnn_kernel = min(k, sum(feature_lengths))
            layer1_cnn_stride = cnn_stride
            layer1_cnn_padding = layer1_cnn_kernel // 2
            layer1_cnn_out_length = math.ceil(
                (feature_lengths[
                     self.text_column_index] + 2 * layer1_cnn_padding - layer1_cnn_kernel + 1) / layer1_cnn_stride)

            layer1_pool_kernel = layer1_cnn_kernel
            layer1_pool_padding = layer1_pool_kernel // 2
            layer1_pool_stride = pool_stride
            layer1_pool_out_length = math.ceil(
                (layer1_cnn_out_length + 2 * layer1_pool_padding - layer1_pool_kernel + 1) / layer1_pool_stride)

            self.logger.info(
                "Cnn layer  out length = {}, layer_cnn_kernel={}, pool layer length = {}, layer_pool_kernel={}".format(
                    layer1_cnn_out_length,
                    layer1_cnn_kernel,
                    layer1_pool_out_length,
                    layer1_pool_kernel
                ))

            layer1 = nn.Sequential(
                nn.Conv1d(total_dim_size, layer1_cnn_output, kernel_size=layer1_cnn_kernel, stride=layer1_cnn_stride,
                          padding=layer1_cnn_padding),
                # nn.BatchNorm1d(layer1_cnn_output),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=layer1_pool_kernel, stride=layer1_pool_stride, padding=layer1_pool_padding))

            self.cnn_layers.append(layer1)
            total_cnn_out_size += layer1_pool_out_length * layer1_cnn_output

        fc_layer_size = 20
        self.fc = nn.Sequential(
            nn.Linear(total_cnn_out_size,
                      fc_layer_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_layer_size, class_size))

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

        text_transposed = text_inputs.transpose(0, 1)

        embeddings = self.embeddings(text_transposed)
        merged_pos_embed = embeddings

        for f in range(len(feature_tuples)):
            if f == self.text_column_index: continue

            entity = feature_tuples[f].transpose(0, 1)

            # TODO: avoid this loop, use builtin
            pos_embedding = []
            for t, e in zip(text_transposed, entity):
                pos_embedding.append(self.pos_embedder(t, e[0]))

            pos_embedding_tensor = torch.stack(pos_embedding)

            merged_pos_embed = torch.cat([merged_pos_embed, pos_embedding_tensor], dim=2)

        # Final output
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        final_input = merged_pos_embed.permute(0, 2, 1)

        outputs = []
        for cnn_layer in self.cnn_layers:
            out1 = cnn_layer(final_input)
            outputs.append(out1)

        out = torch.cat(outputs, dim=2)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
