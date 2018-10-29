from torch import optim, nn

from algorithms.Parser import Parser
from algorithms.PretrainedEmbedderLoader import PretrainedEmbedderLoader
from algorithms.RelationExtractorNetwork import RelationExtractorNetwork
from algorithms.Train import Train


class RelationExtractionFactory:

    def __init__(self, embedding_handle, embedding_dim: int, class_size: int, learning_rate: float = 0.01,
                 momentum: float = 0.9, ngram: int = 3):
        self.ngram = ngram
        self.embedding_dim = embedding_dim
        self.embedding_handle = embedding_handle
        self.class_size = class_size
        self.model_network = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.embedder_loader = None
        self.parser = None
        self.trainer = None
        self.loss_function = None
        self.optimiser = None

    @property
    def model_network(self):
        self.__model_network__ = self.__model_network__ or RelationExtractorNetwork
        return self.__model_network__

    @model_network.setter
    def model_network(self, value):
        self.__model_network__ = value

    @property
    def optimiser(self):
        self.__optimiser__ = self.__optimiser__ or optim.SGD
        return self.__optimiser__

    @optimiser.setter
    def optimiser(self, value):
        self.__optimiser__ = value

    @property
    def loss_function(self):
        self.__loss_function__ = self.__loss_function__ or nn.CrossEntropyLoss()
        return self.__loss_function__

    @loss_function.setter
    def loss_function(self, value):
        self.__loss_function__ = value

    @property
    def trainer(self):
        self.__trainer__ = self.__trainer__ or Train()
        return self.__trainer__

    @trainer.setter
    def trainer(self, value):
        self.__trainer__ = value

    @property
    def parser(self):
        self.__parser__ = self.__parser__ or Parser()
        return self.__parser__

    @parser.setter
    def parser(self, value):
        self.__parser__ = value

    @property
    def embedder_loader(self):
        self.__embedder_loader__ = self.__embedder_loader__ or PretrainedEmbedderLoader()
        return self.__embedder_loader__

    @embedder_loader.setter
    def embedder_loader(self, value):
        self.__embedder_loader__ = value

    def __call__(self, data, labels):
        """

        :type data: Dataframe
        """
        min_words_dict = self.parser.get_min_dictionary()

        # Initialise minwords with random weights
        min_words_weights_dict = {}
        for word in min_words_dict.keys():
            min_words_weights_dict[word] = nn.Embedding(1, self.embedding_dim).weight.detach().numpy().tolist()[0]

        vocab, embedding_array = self.embedder_loader(self.embedding_handle, min_words_weights_dict)

        # Extract words
        data = data.applymap(lambda x: self.parser.split_text(self.parser.normalize_text(x)))

        # TODO Clean this
        model = self.model_network(self.class_size, self.embedding_dim, embedding_array, ngram_context_size=self.ngram)
        processed_data = self.parser.transform_to_array(data.values.tolist(), vocab=vocab)

        #converts labels to int ..
        labels = self.parser.encode_labels(labels)

        data_formatted = [(l, f) for l, f in zip(labels, processed_data)]

        # Set up optimiser
        optimiser = self.optimiser(params=model.parameters(),
                                   lr=self.learning_rate,
                                   momentum=self.momentum)

        # Invoke trainer
        self.trainer(data_formatted, model, self.loss_function, optimiser)
