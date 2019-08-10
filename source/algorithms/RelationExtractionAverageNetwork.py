import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationExtractorNetworkAverage(nn.Module):

    def __init__(self, class_size, embedding_dim, pretrained_weights_or_embed_vocab_size, feature_len,
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
        super(RelationExtractorNetworkAverage, self).__init__()
        # Use random weights if vocab size if passed in else load pretrained weights
        self.embeddings = nn.Embedding(pretrained_weights_or_embed_vocab_size,
                                       embedding_dim) if type(
            pretrained_weights_or_embed_vocab_size) is int else nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_weights_or_embed_vocab_size))
        layer1_size = 128
        # add 2 one for each entity
        self.linear1 = nn.Linear((feature_len) * embedding_dim, layer1_size)
        self.output_layer = nn.Linear(layer1_size, class_size)

    def forward(self, batch_inputs):
        ## Average the embedding words in sentence to represent the sentence embededding
        # index 0 is multiword
        merged_input = []
        for f in batch_inputs:
            concat_sentence = f
            concat_sentence = torch.tensor(concat_sentence, dtype=torch.long)
            embeds = torch.sum(self.embeddings(concat_sentence), dim=1) / len(concat_sentence)
            merged_input.append(embeds)

        final_input = torch.cat(merged_input, dim=1)
        out = F.relu(self.linear1(final_input))
        out = self.output_layer(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
