import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationExtractorLinearNetwork(nn.Module):

    def __init__(self, class_size, embedding_dim, pretrained_weights_or_embed_vocab_size, feature_lengths,
                 ngram_context_size=1,
                 seed=777):
        """
Extracts relationship using a single layer
        :param class_size: The number of relationship types
        :param pretrained_weights_or_embed_vocab_size: Pretrained weights 2 dim array e.g [[.2,.3],[.3,.5]] or the size of vocabulary. When the vocab size if provided,  random embedder is initialised
        :param embedding_dim: The dimension of the embedding
        :param ngram_context_size: The n_gram size
        :param seed: A random seed
        """
        torch.manual_seed(seed)
        super(RelationExtractorLinearNetwork, self).__init__()
        # Use random weights if vocab size if passed in else load pretrained weights

        self.embeddings = nn.Embedding(pretrained_weights_or_embed_vocab_size,
                                       embedding_dim) if type(
            pretrained_weights_or_embed_vocab_size) is int else nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_weights_or_embed_vocab_size))
        layer1_size = 250
        layer2_size = 100
        self.feature_lengths = feature_lengths
        # add 2 one for each entity
        self.linear1 = nn.Linear(sum([f * embedding_dim for f in feature_lengths]), layer1_size)
        self.bn1 = nn.BatchNorm1d(num_features=layer1_size)
        self.linear2 = nn.Linear(layer1_size, layer2_size)
        self.bn2 = nn.BatchNorm1d(num_features=layer2_size)

        self.output_layer = nn.Linear(layer2_size, class_size)

    def forward(self, batch_inputs):
        # Embed each feature
        merged_input = []
        for f, s in zip(batch_inputs, self.feature_lengths):
            concat_sentence = f.transpose(0, 1)
            concat_sentence = torch.tensor(concat_sentence, dtype=torch.long)

            embeddings = self.embeddings(concat_sentence)

            merged_input.append(embeddings)

        # Final output
        final_input = torch.cat(merged_input, dim=1)
        final_input = final_input.view(len(final_input), -1)
        out = F.relu(self.bn1(self.linear1(final_input)))
        out = F.relu(self.bn2(self.linear2(out)))

        out = self.output_layer(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
