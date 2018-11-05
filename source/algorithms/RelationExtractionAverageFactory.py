import itertools
import json
import logging
import os
from collections import Counter

import pandas as pd
from torch import optim, nn
from torchtext import data

from algorithms.Parser import Parser
from algorithms.PretrainedEmbedderLoader import PretrainedEmbedderLoader
from algorithms.RelationExtractionAverageNetwork import RelationExtractorNetworkAverage
from algorithms.Train import Train


class RelationExtractionAverageFactory:

    def __init__(self, embedding_handle, embedding_dim: int, class_size: int, output_dir, learning_rate: float = 0.01,
                 momentum: float = 0.9, ngram: int = 3, epochs: int = 10):
        self.epochs = epochs
        self.output_dir = output_dir
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
        self.__model_network__ = self.__model_network__ or RelationExtractorNetworkAverage
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

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @embedder_loader.setter
    def embedder_loader(self, value):
        self.__embedder_loader__ = value

    def __call__(self, train, train_labels, validation, validation_labels):
        """

        :type data: Dataframe
        """
        min_words_dict = self.parser.get_min_dictionary()

        # Initialise minwords with random weights
        min_words_weights_dict = {}
        for word in min_words_dict.keys():
            min_words_weights_dict[word] = nn.Embedding(1, self.embedding_dim).weight.detach().numpy().tolist()[0]

        self.logger.info("Loading embeding..")
        vocab, embedding_array = self.embedder_loader(self.embedding_handle, min_words_weights_dict)
        self.logger.info("loaded vocab size {}, embed array len {}, size of first element {}.".format(len(vocab), len(
            embedding_array), len(embedding_array[0])))

        # TODO: expecting first column to be an abstract so the network averages the sentence
        self.col_names = train.columns.values

        # Extract words
        train_data = train.applymap(lambda x: self.parser.split_text(self.parser.normalize_text(x)))
        validation_data = validation.applymap(lambda x: self.parser.split_text(self.parser.normalize_text(x)))

        # TODO Clean this
        model = self.model_network(self.class_size, self.embedding_dim, embedding_array,
                                   feature_len=len(self.col_names))
        processed_data = self.parser.transform_to_array(train_data.values.tolist(), vocab=vocab)
        val_processed_data = self.parser.transform_to_array(validation_data.values.tolist(), vocab=vocab)

        token_counts = pd.DataFrame(processed_data).apply(lambda c: self.get_column_values_count(c), axis=0).values
        self.logger.info("Token counts : {}".format(token_counts))

        # converts train_labels_encode to int ..
        classes = self.parser.get_label_map(train_labels)
        train_labels_encode = self.parser.encode_labels(train_labels, classes)
        validation_labels_encode = self.parser.encode_labels(validation_labels, classes)

        data_formatted, val_data_formatted = self.getexamples(self.col_names, processed_data,
                                                              train_labels_encode), self.getexamples(
            self.col_names, val_processed_data,
            validation_labels_encode)

        sort_key = lambda x: self.sum(x)

        # Set up optimiser
        optimiser = self.optimiser(params=model.parameters(),
                                   lr=self.learning_rate,
                                   momentum=self.momentum)

        self.persist(outdir=self.output_dir, vocab=vocab)

        # Invoke trainer
        self.trainer(data_formatted, val_data_formatted, sort_key, model, self.loss_function, optimiser,
                     self.output_dir, epoch=self.epochs)

    def get_column_values_count(self, c):
        values = list(itertools.chain.from_iterable(c.values))
        return Counter(values)

    def sum(self, x):
        return sum([len(getattr(x, c)) for c in self.col_names])

    def persist(self, outdir, vocab):
        with open(os.path.join(outdir, "vocab.json"), "w") as f:
            f.write(json.dumps(vocab))

    def getexamples(self, col_names, data_list, encoded_labels):
        LABEL = data.LabelField(use_vocab=False, sequential=False, is_target=True)

        fields = [(c, data.Field(use_vocab=False, pad_token=0)) for c in col_names]
        fields.append(("label", LABEL))

        return data.Dataset([data.Example.fromlist([*f, l], fields) for l, f in
                             zip(encoded_labels, data_list)], fields)
