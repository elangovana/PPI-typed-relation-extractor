import glob
import logging
import os
import pickle

import torch
from keras.optimizers import RMSprop
from torch.utils.data import DataLoader

from algorithms.Collator import Collator
from algorithms.Predictor import Predictor
from algorithms.VocabMerge import VocabMerger


class TrainInferencePipeline:

    def __init__(self, model, loss_function, trainer, train_vocab_extractor, embedder_loader,
                 embedding_handle, embedding_dim: int,
                 label_pipeline, data_pipeline, class_size: int, pos_label, model_dir, output_dir, ngram: int = 3,
                 min_vocab_frequency=3, class_weights_dict=None, num_workers=None, batch_size=32, additional_args=None,
                 merge_train_val_vocab=False):
        self.merge_train_val_vocab = merge_train_val_vocab
        self.additional_args = additional_args or {}
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = 1 if os.cpu_count() == 1 else max(int(os.cpu_count() / 4), 1)

        self.trainer = trainer
        self.class_weights_dict = class_weights_dict
        self.embedding_handle = embedding_handle
        self.embedder_loader = embedder_loader
        self.loss_function = loss_function
        self.model = model
        self.data_pipeline = data_pipeline
        self.label_pipeline = label_pipeline

        self.pos_label = pos_label
        self.min_vocab_frequency = min_vocab_frequency
        self.output_dir = output_dir
        self.ngram = ngram
        self.embedding_dim = embedding_dim
        self.class_size = class_size
        self.train_vocab_extractor = train_vocab_extractor

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self.logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

    def __call__(self, train, validation):
        self.logger.info("Train set has {} records, val has {}".format(len(train), len(validation)))
        train_loader = DataLoader(train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,
                                  collate_fn=Collator())
        val_loader = DataLoader(validation, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers,
                                collate_fn=Collator())

        # Merge train vocab and the pretrained vocab
        self.embedding_handle.seek(0)
        train_vocab_dict = self.train_vocab_extractor.construct_vocab_dict(train_loader)

        if self.merge_train_val_vocab:
            val_dict = self.train_vocab_extractor.construct_vocab_dict(val_loader)
            train_vocab_dict = VocabMerger()(train_vocab_dict, val_dict)

        full_vocab_dict, embedding_array = self.embedder_loader(self.embedding_handle, train_vocab_dict)
        self.data_pipeline.update_vocab_dict(full_vocab_dict)

        transformed_train_x = self.data_pipeline.fit_transform(train_loader)
        transformed_val_x = self.data_pipeline.transform(val_loader)

        transformed_train_x = self.label_pipeline.fit_transform(transformed_train_x)
        transformed_val_x = self.label_pipeline.transform(transformed_val_x)

        tensor_embeddings = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_array))
        self.model.set_embeddings(tensor_embeddings)

        # Optimiser
        learning_rate = float(self._get_value(self.additional_args, "learningrate", ".01"))

        # optimiser = SGD(lr=self.learning_rate, momentum=self.momentum, params=model.parameters())
        weight_decay = float(self._get_value(self.additional_args, "weight_decay", ".0001"))
        # optimiser = Adam(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimiser = RMSprop(params=self.model.parameters(), lr=learning_rate)
        self.logger.info("Using optimiser {}".format(type(optimiser)))
        # Set weights
        # if self.class_weights_dict is not None:
        #     self._class_weights = [1] * len(classes)
        #     for k, w in self.class_weights_dict.items():
        #         class_int = transformer_labels.transform([k])[0]
        #         self._class_weights[class_int] = w
        #     self._class_weights = torch.Tensor(self._class_weights)
        #     self.logger.info("Class weights dict is : {}".format(self.class_weights_dict))
        #     self.logger.info("Class weights are is : {}".format(self._class_weights))

        # Lengths of each column

        encoded_pos_label = self.label_pipeline.transform(self.pos_label)
        self.logger.info("Positive label {} is {}".format(self.pos_label, encoded_pos_label))

        # Set up optimiser

        self.persist(outdir=self.model_dir)

        # Invoke trainer
        (val_results, val_actuals, val_predicted) = self.trainer(transformed_train_x, transformed_val_x,

                                                                 self.model, self.loss_function,
                                                                 optimiser, self.model_dir,
                                                                 self.output_dir, pos_label=encoded_pos_label)

        # Reformat results so that the labels are back into their original form, rather than numbers
        val_actuals = self.label_pipeline.label_reverse_encoder_func(val_actuals)
        val_predicted = self.label_pipeline.label_reverse_encoder_func(val_predicted)

        return val_results, val_actuals, val_predicted

    def sum(self, x):
        return sum([len(getattr(x, c)) for c in x.__dict__ if c != 'label'])

    def persist(self, outdir):
        with open(os.path.join(outdir, "picked_datapipeline.pb"), "wb") as f:
            pickle.dump(self.data_pipeline, f)

        with open(os.path.join(outdir, "picked_labelpipeline.pb"), "wb") as f:
            pickle.dump(self.label_pipeline, f)

    @staticmethod
    def load(artifacts_dir):
        model_file = TrainInferencePipeline._find_artifact("{}/*model.pt".format(artifacts_dir))

        data_pipeline = TrainInferencePipeline._load_artifact("{}/*picked_datapipeline.pb".format(artifacts_dir))
        label_pipeline = TrainInferencePipeline._load_artifact("{}/*picked_labelpipeline.pb".format(artifacts_dir))

        model = torch.load(model_file)

        return lambda x: TrainInferencePipeline.predict(x, model, data_pipeline, label_pipeline)

    @staticmethod
    def _load_artifact(pickled_file_search_filter):
        datapipeline_file = TrainInferencePipeline._find_artifact(pickled_file_search_filter)
        with open(datapipeline_file, "rb") as f:
            datapipeline = pickle.load(f)

        return datapipeline

    @staticmethod
    def _find_artifact(pattern):
        matching = glob.glob(pattern)
        assert len(matching) == 1, "Expected exactly one in {}, but found {}".format(pattern,
                                                                                     len(matching))
        matched_file = matching[0]
        return matched_file

    @staticmethod
    def predict(dataset, model, data_pipeline, label_pipeline):
        dataloader = DataLoader(dataset, shuffle=False, batch_size=32, num_workers=1,
                                collate_fn=Collator())

        val_examples = data_pipeline.transform(dataloader)

        predictor = Predictor()

        predictions, confidence_scores = predictor.predict(model, val_examples)

        transformed_predictions = label_pipeline.label_reverse_encoder_func(predictions)

        transformed_conf_scores = TrainInferencePipeline._get_confidence_score_dict(label_pipeline, confidence_scores)

        return transformed_predictions, transformed_conf_scores

    @staticmethod
    def _get_confidence_score_dict(label_pipeline, confidence_scores):
        transformed_conf_scores = []
        for b in confidence_scores:
            for r in b:
                conf_score = {}
                for i, s in enumerate(r):
                    conf_score[label_pipeline.label_reverse_encoder_func(i)] = s
                transformed_conf_scores.append(conf_score)

        return transformed_conf_scores
