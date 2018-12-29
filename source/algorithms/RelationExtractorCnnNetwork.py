import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationExtractorCnnNetwork(nn.Module):

    def __init__(self, class_size, embedding_dim, pretrained_weights_or_embed_vocab_size, feature_lengths,
                 ngram_context_size=5, seed=777, drop_rate=.3):
        self.feature_lengths = feature_lengths
        torch.manual_seed(seed)

        super(RelationExtractorCnnNetwork, self).__init__()
        self.logger.info("NGram Size is {}".format(ngram_context_size))
        self.dropout_rate = drop_rate
        # Use random weights if vocab size if passed in else load pretrained weights

        self.embeddings = nn.Embedding(pretrained_weights_or_embed_vocab_size,
                                       embedding_dim) if type(
            pretrained_weights_or_embed_vocab_size) is int else nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_weights_or_embed_vocab_size))
        layer1_cnn_output = 128
        layer1_cnn_kernel = min(ngram_context_size, sum(feature_lengths))
        layer1_cnn_stride = 1
        layer1_cnn_padding = 0
        layer1_cnn_out_length = math.ceil(
            (sum(feature_lengths) + 2 * layer1_cnn_padding - layer1_cnn_kernel + 1) / layer1_cnn_stride)

        layer1_pool_kernel = 2
        layer1_pool_padding = 1
        layer1_pool_stride = 2
        layer1_pool_out_length = math.ceil(
            (layer1_cnn_out_length + 2 * layer1_pool_padding - layer1_pool_kernel + 1) / layer1_pool_stride)

        self.logger.info("Cnn layer out length = {}, pool layer length = {}".format(layer1_cnn_out_length,
                                                                                    layer1_pool_out_length))

        self.layer1 = nn.Sequential(
            nn.Conv1d(embedding_dim, layer1_cnn_output, kernel_size=layer1_cnn_kernel, stride=layer1_cnn_stride,
                      padding=layer1_cnn_padding),
            nn.BatchNorm1d(layer1_cnn_output),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=layer1_pool_kernel, stride=layer1_pool_stride, padding=layer1_pool_padding))

        layer2_cnn_output = 128
        layer2_cnn_kernel = min(abs(layer1_cnn_kernel - 2), 1)
        layer2_cnn_stride = 1
        layer2_cnn_padding = 0
        layer2_cnn_out_length = math.ceil(
            (layer1_pool_out_length + 2 * layer2_cnn_padding - layer2_cnn_kernel + 1) / layer2_cnn_stride)

        layer2_pool_kernel = 2
        layer2_pool_padding = 1
        layer2_pool_stride = 2
        layer2_pool_out_length = math.ceil(
            (layer2_cnn_out_length + 2 * layer2_pool_padding - layer2_pool_kernel + 1) / layer2_pool_stride)
        self.layer2 = nn.Sequential(
            nn.Conv1d(layer1_cnn_output, layer2_cnn_output, kernel_size=layer2_cnn_kernel, stride=layer2_cnn_stride,
                      padding=layer2_cnn_padding),
            nn.BatchNorm1d(layer2_cnn_output),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=layer2_pool_kernel, stride=layer2_pool_stride, padding=layer2_pool_padding))

        fc_layer_size = 100
        self.fc = nn.Sequential(nn.Linear(layer2_pool_out_length * layer2_cnn_output, fc_layer_size), nn.ReLU(),
                                nn.Linear(fc_layer_size, class_size))

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def forward(self, batch_inputs):
        merged_input = []
        # Get longest feature
        # Assume the longest column is the
        max_words = max(self.feature_lengths)
        for f, feature_len in zip(batch_inputs, self.feature_lengths):
            concat_sentence = f.transpose(0, 1)
            concat_sentence = torch.tensor(concat_sentence, dtype=torch.long)

            embeddings = self.embeddings(concat_sentence)
            # if feature_len == max_words:
            #     # Set up success rate (rate of selecting the word) as 1 - dropout rate
            #     bernoulli = Bernoulli(1 - self.dropout_rate)
            #     rw = bernoulli.sample(torch.Size((embeddings.shape[0], embeddings.shape[1]))).numpy()
            #     # Use zeros at where rw is zero
            #     embeddings = torch.from_numpy(np.expand_dims(rw, 2)) * embeddings

            merged_input.append(embeddings)

        # Final output
        final_input = torch.cat(merged_input, dim=1)
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        final_input = final_input.permute(0, 2, 1)

        out = self.layer1(final_input)

        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
