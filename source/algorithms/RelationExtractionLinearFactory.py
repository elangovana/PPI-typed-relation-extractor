import glob
import json
import logging
import os

import numpy
import torch
from sklearn.pipeline import Pipeline
from torch import optim, nn

from algorithms.PretrainedEmbedderLoader import PretrainedEmbedderLoader
from algorithms.RelationExtractorLinearNetwork import RelationExtractorLinearNetwork
from algorithms.Train import Train
from algorithms.VocabRandomPretrainedEmbedCombiner import VocabRandomPretrainedEmbedCombiner
from algorithms.transform_extract_label_numbers import TransformExtractLabelNumbers
from algorithms.transform_final_create_examples import TransformFinalCreateExamples
from algorithms.transform_labels_to_numbers import TransformLabelsToNumbers
from algorithms.transform_tokenise import TransformTokenise
from algorithms.transform_tokens_to_indices import TransformTokensToIndices


class RelationExtractionLinearFactory:

    def __init__(self, embedding_handle, embedding_dim: int, class_size: int, output_dir, learning_rate: float = 0.01,
                 momentum: float = 0.9, ngram: int = 3, epochs: int = 10, min_vocab_frequency=3, pos_label=1,
                 classes=None):
        self.classes = classes
        self.pos_label = pos_label
        self.min_vocab_frequency = min_vocab_frequency
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
        self.transform_extract_label_number = None
        self.transform_tokenise = None
        self.train_data_pipeline = None
        self.vocab_embedding_builder = None

    @property
    def model_network(self):
        self.__model_network__ = self.__model_network__ or RelationExtractorLinearNetwork
        return self.__model_network__

    @model_network.setter
    def model_network(self, value):
        self.__model_network__ = value

    @property
    def optimiser(self):
        self.__optimiser__ = self.__optimiser__ or optim.RMSprop
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
    def embedder_loader(self):
        self.__embedder_loader__ = self.__embedder_loader__ or PretrainedEmbedderLoader()
        return self.__embedder_loader__

    @embedder_loader.setter
    def embedder_loader(self, value):
        self.__embedder_loader__ = value

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def get_data_pipeline(self, vocab, pipeline=None):
        # this is the default pipeline

        pipeline = pipeline or Pipeline([
            ('TransformExtractWords', self.transform_tokenise)
            , ('TransformWordsIndices', TransformTokensToIndices(vocab=vocab))

        ])

        return pipeline

    @property
    def transform_extract_label_number(self):
        self.__transform_extract_label_number__ = self.__transform_extract_label_number__ or TransformExtractLabelNumbers()
        return self.__transform_extract_label_number__

    @transform_extract_label_number.setter
    def transform_extract_label_number(self, value):
        self.__transform_extract_label_number__ = value

    @property
    def vocab_embedding_builder(self):
        self.__vocab_embedding_builder__ = self.__vocab_embedding_builder__ or VocabRandomPretrainedEmbedCombiner(
            self.embedding_handle)
        return self.__vocab_embedding_builder__

    @vocab_embedding_builder.setter
    def vocab_embedding_builder(self, value):
        self.__vocab_embedding_builder__ = value

    @property
    def transform_tokenise(self):
        self.__transform_tokenise__ = self.__transform_tokenise__ or TransformTokenise()
        return self.__transform_tokenise__

    @transform_tokenise.setter
    def transform_tokenise(self, value):
        self.__transform_tokenise__ = value

    def get_transform_examples(self, feature_len, transformer=None):
        transformer = transformer or TransformFinalCreateExamples(feature_lens=feature_len)
        return transformer

    def get_transformer_labels_to_integers(self, classes, transformer=None):
        transformer = transformer or TransformLabelsToNumbers(classes=classes)

        return transformer

    def __call__(self, train, train_labels, validation, validation_labels):
        full_vocab, embedding_array = self.vocab_embedding_builder(train)

        self.logger.info(
            "loaded vocab size {}, embed array len {}, size of first element {}.".format(len(full_vocab), len(
                embedding_array), len(embedding_array[0])))

        classes = self.transform_extract_label_number.transform(train_labels)

        transformer_labels = self.get_transformer_labels_to_integers(classes)
        transfomed_train_labels = transformer_labels.transform(train_labels)
        transfomed_val_labels = transformer_labels.transform(validation_labels)

        self.logger.debug("Transformed train labels : {}".format(transfomed_train_labels))
        self.logger.debug("Transformed val labels : {}".format(transfomed_val_labels))

        transformer_pipeline = self.get_data_pipeline(vocab=full_vocab)
        # Lengths of each column
        feature_lens = transformer_pipeline.transform(train).apply(
            lambda c: numpy.math.ceil(numpy.percentile(c.apply(len), 90))).values
        self.logger.info("Column length counts : {}".format(feature_lens))
        transformer_examples = self.get_transform_examples(feature_lens)

        train_examples = transformer_examples.transform(transformer_pipeline.transform(train),
                                                        y=transfomed_train_labels)
        val_examples = transformer_examples.transform(transformer_pipeline.transform(validation),
                                                      y=transfomed_val_labels)

        model = self.model_network(self.class_size, self.embedding_dim, embedding_array,
                                   feature_lengths=feature_lens)

        pos_label = transformer_labels.encode_labels([self.pos_label], classes)[0]

        sort_key = lambda x: self.sum(x)

        # Set up optimiser
        optimiser = self.optimiser(params=model.parameters(),
                                   lr=self.learning_rate,
                                   momentum=self.momentum)

        self.persist(outdir=self.output_dir, vocab=full_vocab, classes=classes, feature_lens=feature_lens)

        # Invoke trainer
        (model_network, val_results, val_actuals, val_predicted) = self.trainer(train_examples, val_examples, sort_key,
                                                                                model, self.loss_function, optimiser,
                                                                                self.output_dir, epoch=self.epochs,
                                                                                pos_label=pos_label)

        # Reformat results so that the labels are back into their original form, rather than numbers
        val_actuals = [classes[p] for p in val_actuals]
        val_predicted = [classes[p] for p in val_predicted]

        return model_network, val_results, val_actuals, val_predicted

    def sum(self, x):
        return sum([len(getattr(x, c)) for c in x.__dict__ if c != 'label'])

    def persist(self, outdir, vocab, classes, feature_lens):
        with open(os.path.join(outdir, "vocab.json"), "w") as f:
            f.write(json.dumps(vocab))

        with open(os.path.join(outdir, "classes.json"), "w") as f:
            f.write(json.dumps(classes.tolist()))

        with open(os.path.join(outdir, "feature_lens.json"), "w") as f:
            f.write(json.dumps(feature_lens.tolist()))

    @staticmethod
    def load(artifacts_dir):
        model_file = RelationExtractionLinearFactory._find_artifact("{}/*model.pt".format(artifacts_dir))
        classes_file = RelationExtractionLinearFactory._find_artifact("{}/*classes.json".format(artifacts_dir))
        vocab_file = RelationExtractionLinearFactory._find_artifact("{}/*vocab.json".format(artifacts_dir))
        feature_lens_file = RelationExtractionLinearFactory._find_artifact(
            "{}/*feature_lens.json".format(artifacts_dir))

        with open(vocab_file, "r") as f:
            vocab = json.loads(f.read())

        with open(feature_lens_file, "r") as f:
            feature_lens = numpy.asarray(json.loads(f.read()))

        with open(classes_file, "r") as f:
            classes = numpy.asarray(json.loads(f.read()))

        factory = RelationExtractionLinearFactory(class_size=0, embedding_handle=None, output_dir=None,
                                                  embedding_dim=0)

        model = torch.load(model_file)

        return lambda x: factory.predict(x, vocab, feature_lens, model, classes)

    @staticmethod
    def _find_artifact(pattern):
        matching = glob.glob(pattern)
        assert len(matching) == 1, "Expected exactly one in {}, but found {}".format(pattern,
                                                                                     len(matching))
        matched_file = matching[0]
        return matched_file

    def predict(self, df, vocab, feature_lens, model, classes):
        self.logger.debug("Vocab: \n{}".format(vocab))

        transformer_pipeline = self.get_data_pipeline(vocab=vocab)
        transformer_examples = self.get_transform_examples(feature_lens)

        val_examples = transformer_examples.transform(transformer_pipeline.transform(df))

        predictions = self.trainer.predict(model, val_examples)

        self.logger.debug("Predictions_raw \n{}".format(predictions))

        transformed_predictions = [classes[p] for p in predictions]

        self.logger.debug("Predictions Transformed \n{}".format(transformed_predictions))

        return transformed_predictions
