from algorithms.transform_text_index import TransformTextToIndex


class DataPipeline:

    def __init__(self, pretrained_embedder_loader, embeddings_handle, max_feature_lens):
        self.max_feature_lengths = max_feature_lens
        self.vocab = None
        self._vocab_index, _ = pretrained_embedder_loader(embeddings_handle)

    @property
    def text_to_index(self):
        self.__text_to_index__ = getattr(self, "__text_to_index__", None) or TransformTextToIndex(
            vocab=self.vocab, max_feature_lens=self.max_feature_lengths)
        return self.__text_to_index__

    @text_to_index.setter
    def text_to_index(self, value):
        self.__text_to_index__ = value

    def transform(self, dataloader):
        transformed_x = dataloader
        for name, p in self.feature_pipeline:
            transformed_x = p.transform(transformed_x)
        return transformed_x

    def fit_transform(self, dataloader):
        self.fit(dataloader)
        return self.transform(dataloader)

    def fit(self, dataloader):
        # Load pretrained vocab
        vocab_index = self._vocab_index
        self.vocab = [None] * len(vocab_index)
        for k, v in vocab_index.items():
            self.vocab[v] = k

        # set up pipeline
        self.feature_pipeline = [("text_to_index", self.text_to_index)]

        # load count vectoriser after loading pretrained vocab
        for name, p in self.feature_pipeline:
            p.fit(dataloader)

        # update vocab after fit
        self.vocab = self.text_to_index.vocab
