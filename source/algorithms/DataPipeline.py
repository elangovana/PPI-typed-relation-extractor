from sklearn.pipeline import Pipeline

from algorithms.PretrainedEmbedderLoader import PretrainedEmbedderLoader
from algorithms.transform_final_create_examples import TransformFinalCreateExamples
from algorithms.transform_label_encoder import TransformLabelEncoder
from algorithms.transform_text_index import TransformTextToIndex


class DataPipeline:

    def __init__(self, embeddings_handle=None):
        self.vocab = None
        self.embeddings_handle = embeddings_handle
        self.label_pipeline = None

    @property
    def pretrained_embedder_loader(self):
        self.__pretrained_embedder_loader__ = getattr(self, "__pretrained_embedder_loader__",
                                                      None) or PretrainedEmbedderLoader()
        return self.__pretrained_embedder_loader__

    @pretrained_embedder_loader.setter
    def pretrained_embedder_loader(self, value):
        self.__pretrained_embedder_loader__ = value

    @property
    def label_encoder(self):
        self.__label_encoder__ = getattr(self, "__label_encoder__", None) or TransformLabelEncoder()
        return self.__label_encoder__

    @label_encoder.setter
    def label_encoder(self, value):
        self.__label_encoder__ = value

    @property
    def label_reverse_encoder_func(self):
        return self.label_encoder.inverse_transform

    @property
    def count_vectoriser(self):
        self.__count_vectoriser__ = getattr(self, "__count_vectoriser__", None) or TransformTextToIndex(
            vocab=self.vocab)
        return self.__count_vectoriser__

    @count_vectoriser.setter
    def count_vectoriser(self, value):
        self.__count_vectoriser__ = value

    @property
    def example_transformer(self):
        self.__example_transformer__ = getattr(self, "__example_transformer__", None) or TransformFinalCreateExamples()
        return self.__example_transformer__

    @example_transformer.setter
    def example_transformer(self, value):
        self.__example_transformer__ = value

    def transform(self, data_x, data_y=None):
        transformed_y = self.label_pipeline.transform(data_y)
        transformed_x = self.feature_pipeline.transform(data_x)
        return transformed_x, transformed_y

    def fit_transform(self, data_x, data_y):
        self.fit(data_x, data_y)
        return self.transform(data_x, data_y)

    def fit(self, data_x, data_y):
        # Load pretrained vocab
        vocab_index, _ = self.pretrained_embedder_loader(self.embeddings_handle)
        self.vocab = [None] * len(vocab_index)
        for k, v in vocab_index.items():
            self.vocab[v] = k

        # set up pipeline
        self.feature_pipeline = Pipeline(
            steps=[("count_vector", self.count_vectoriser), ("example_transformer", self.example_transformer)])

        self.label_pipeline = Pipeline(steps=[("label_encoder", self.label_encoder)])

        # load count vectoriser after loading pretrained vocab
        self.label_pipeline.fit(data_y, None)
        self.feature_pipeline.fit(data_x, data_y)
