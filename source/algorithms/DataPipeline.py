from sklearn.pipeline import Pipeline

from algorithms.transform_text_index import TransformTextToIndex


class DataPipeline:

    def __init__(self, pretrained_embedder_loader, embeddings_handle, max_feature_lens):
        self.max_feature_lengths = max_feature_lens
        self.pretrained_embedder_loader = pretrained_embedder_loader
        self.vocab = None
        self.embeddings_handle = embeddings_handle

    @property
    def count_vectoriser(self):
        self.__count_vectoriser__ = getattr(self, "__count_vectoriser__", None) or TransformTextToIndex(
            vocab=self.vocab, max_feature_lens=self.max_feature_lengths)
        return self.__count_vectoriser__

    @count_vectoriser.setter
    def count_vectoriser(self, value):
        self.__count_vectoriser__ = value

    def transform(self, dataloader):
        transformed_x = self.feature_pipeline.transform(dataloader)
        return transformed_x

    def fit_transform(self, dataloader):
        self.fit(dataloader)
        return self.transform(dataloader)

    def fit(self, dataloader):
        # Load pretrained vocab
        self.embeddings_handle.seek(0)
        vocab_index, _ = self.pretrained_embedder_loader(self.embeddings_handle)
        self.vocab = [None] * len(vocab_index)
        for k, v in vocab_index.items():
            self.vocab[v] = k

        # set up pipeline
        self.feature_pipeline = Pipeline(
            steps=[("count_vector", self.count_vectoriser)])

        # load count vectoriser after loading pretrained vocab
        for name, p in self.feature_pipeline.steps:
            p.fit(dataloader)

        # update vocab after fit
        self.vocab = self.count_vectoriser.vocab
