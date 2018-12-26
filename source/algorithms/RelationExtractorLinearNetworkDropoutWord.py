import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.bernoulli import Bernoulli


class RelationExtractorLinearNetworkDropoutWord(nn.Module):

    def __init__(self, class_size, embedding_dim, pretrained_weights_or_embed_vocab_size, feature_lengths,
                 ngram_context_size=1,
                 seed=777, dropout_rate=.3):
        """
Extracts relationship using a single layer
        :param class_size: The number of relationship types
        :param pretrained_weights_or_embed_vocab_size: Pretrained weights 2 dim array e.g [[.2,.3],[.3,.5]] or the size of vocabulary. When the vocab size if provided,  random embedder is initialised
        :param embedding_dim: The dimension of the embedding
        :param ngram_context_size: The n_gram size
        :param seed: A random seed
        """
        self.dropout_rate = dropout_rate
        torch.manual_seed(seed)
        super(RelationExtractorLinearNetworkDropoutWord, self).__init__()
        # Use random weights if vocab size if passed in else load pretrained weights

        self.embeddings = nn.Embedding(pretrained_weights_or_embed_vocab_size,
                                       embedding_dim) if type(
            pretrained_weights_or_embed_vocab_size) is int else nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_weights_or_embed_vocab_size))
        layer1_size = 100
        layer2_size = 70
        layer3_size = 40
        layer4_size = 20
        layer5_size = 10
        layer6_size = 5
        layer7_size = 5

        self.feature_lengths = feature_lengths
        # add 2 one for each entity
        self.linear1 = nn.Linear(sum([f * embedding_dim for f in feature_lengths]), layer1_size)
        self.linear2 = nn.Linear(layer1_size, layer2_size)
        self.linear3 = nn.Linear(layer2_size, layer3_size)
        self.linear4 = nn.Linear(layer3_size, layer4_size)
        self.linear5 = nn.Linear(layer4_size, layer5_size)
        self.linear6 = nn.Linear(layer5_size, layer6_size)
        self.linear7 = nn.Linear(layer6_size, layer7_size)

        self.output_layer = nn.Linear(layer7_size, class_size)

    def add_unk(input_token_id, p):
        # random.random() gives you a value between 0 and 1
        # to avoid switching your padding to 0 we add 'input_token_id > 1'
        if torch.random.random() < p and input_token_id > 1:
            return 0
        else:
            return input_token_id

    def forward(self, batch_inputs):
        # Embed each feature
        merged_input = []
        # Get longest feature
        # Assume the longest column is the
        max_words = max(self.feature_lengths)

        for input, feature_len in zip(batch_inputs, self.feature_lengths):
            concat_sentence = input.transpose(0, 1)
            concat_sentence = torch.tensor(concat_sentence, dtype=torch.long)

            embeddings = self.embeddings(concat_sentence)

            if feature_len == max_words:
                # Set up success rate (rate of selecting the word) as 1 - dropout rate
                bernoulli = Bernoulli(1 - self.dropout_rate)
                rw = bernoulli.sample(torch.Size((embeddings.shape[0], embeddings.shape[1]))).numpy()
                # Use zeros at where rw is zero
                embeddings = torch.from_numpy(np.expand_dims(rw, 2)) * embeddings

            merged_input.append(embeddings)

        # Final output
        final_input = torch.cat(merged_input, dim=1)
        final_input = final_input.view(len(final_input), -1)
        out = torch.tanh(self.linear1(final_input))
        out = torch.tanh(self.linear2(out))
        out = torch.tanh(self.linear3(out))
        out = torch.tanh(self.linear4(out))
        out = torch.tanh(self.linear5(out))
        out = torch.tanh(self.linear6(out))

        out = F.relu(self.linear7(out))

        out = self.output_layer(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
