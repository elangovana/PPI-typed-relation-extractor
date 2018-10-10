import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationExtractorNetwork(nn.Module):

    def __init__(self, class_size, embedding_dim, pretrained_weights_or_embed_vocab_size,
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
        super(RelationExtractorNetwork, self).__init__()
        # Use random weights if vocab size if passed in else load pretrained weights
        self.embeddings = nn.Embedding(pretrained_weights_or_embed_vocab_size,
                                       pretrained_weights_or_embed_vocab_size) if pretrained_weights_or_embed_vocab_size is int else nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_weights_or_embed_vocab_size))
        layer1_size = 128
        self.linear1 = nn.Linear(ngram_context_size * embedding_dim, layer1_size)
        self.output_layer = nn.Linear(layer1_size, class_size)

    def forward(self, inputs):
        ## Average the embedding words in sentence to represent the sentence embededding
        embeds = torch.sum(self.embeddings(inputs), dim=1) / len(inputs)
        out = F.relu(self.linear1(embeds))
        out = self.output_layer(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
