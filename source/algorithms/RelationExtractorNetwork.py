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
        # add 2 one for each entity
        self.linear1 = nn.Linear((ngram_context_size + 2) * embedding_dim, layer1_size)
        self.output_layer = nn.Linear(layer1_size, class_size)

    def forward(self, batch_inputs):
        ## Average the embedding words in sentence to represent the sentence embededding
        # index 0 is multiword
        concat_sentence = batch_inputs[0].transpose(0, 1)
        concat_sentence = torch.tensor(concat_sentence, dtype=torch.long)
        embeds = torch.sum(self.embeddings(concat_sentence), dim=1) / len(concat_sentence)

        concat_entity1 = batch_inputs[1].transpose(0, 1)
        concat_entity1 = torch.tensor(concat_entity1, dtype=torch.long)
        embde_entity1 = torch.sum(self.embeddings(concat_entity1), dim=1) / len(concat_entity1)

        concat_entity2 = batch_inputs[2].transpose(0, 1)
        concat_entity2 = torch.tensor(concat_entity2, dtype=torch.long)
        embde_entity2 = torch.sum(self.embeddings(concat_entity2), dim=1) / len(concat_entity2)

        final_input = torch.cat([embeds, embde_entity1, embde_entity2], dim=1)
        out = F.relu(self.linear1(final_input))
        out = self.output_layer(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
