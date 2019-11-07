import glob
import logging
import os
import pickle

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from algorithms.Collator import Collator
from algorithms.Predictor import Predictor


class BertTrainInferencePipeline:

    def __init__(self, model, loss_function, trainer,
                 label_pipeline, data_pipeline, class_size: int, pos_label, model_dir, output_dir, ngram: int = 3,
                 min_vocab_frequency=3, class_weights_dict=None, num_workers=None, batch_size=32, additional_args=None):
        self.additional_args = additional_args or {}
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = 1 if os.cpu_count() == 1 else max(int(os.cpu_count() / 4), 1)

        self.trainer = trainer
        self.class_weights_dict = class_weights_dict

        self.loss_function = loss_function
        self.model = model
        self.data_pipeline = data_pipeline
        self.label_pipeline = label_pipeline

        self.pos_label = pos_label
        self.min_vocab_frequency = min_vocab_frequency
        self.output_dir = output_dir
        self.ngram = ngram
        self.class_size = class_size

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

        transformed_train_x = self.data_pipeline.fit_transform(train_loader)
        transformed_val_x = self.data_pipeline.transform(val_loader)

        transformed_train_x = self.label_pipeline.fit_transform(transformed_train_x)
        transformed_val_x = self.label_pipeline.transform(transformed_val_x)

        # Optimiser
        learning_rate = float(self._get_value(self.additional_args, "learningrate", ".01"))
        # optimiser = SGD(lr=self.learning_rate, momentum=self.momentum, params=model.parameters())
        weight_decay = float(self._get_value(self.additional_args, "weight_decay", ".0001"))
        optimiser = Adam(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # optimiser = RMSprop(params=model.parameters(), lr=learning_rate)
        self.logger.info("Using optimiser {}".format(type(optimiser)))

        # Lengths of each column
        encoded_pos_label = self.label_pipeline.transform(self.pos_label)
        self.logger.info("Positive label {} is {}".format(self.pos_label, encoded_pos_label))

        # Set up optimiser

        self.persist(outdir=self.model_dir)

        # Invoke trainer
        (val_results, val_actuals, val_predicted) = self.trainer(transformed_train_x, transformed_val_x,

                                                                 self.model, self.loss_function,
                                                                 optimiser, self.model_dir, self.output_dir,
                                                                 pos_label=encoded_pos_label)

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
        model_file = BertTrainInferencePipeline._find_artifact("{}/*model.pt".format(artifacts_dir))

        data_pipeline = BertTrainInferencePipeline._load_artifact("{}/*picked_datapipeline.pb".format(artifacts_dir))
        label_pipeline = BertTrainInferencePipeline._load_artifact("{}/*picked_labelpipeline.pb".format(artifacts_dir))

        model = torch.load(model_file)

        return lambda x: BertTrainInferencePipeline.predict(x, model, data_pipeline, label_pipeline)

    @staticmethod
    def _load_artifact(pickled_file_search_filter):
        datapipeline_file = BertTrainInferencePipeline._find_artifact(pickled_file_search_filter)
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

        transformed_conf_scores = BertTrainInferencePipeline._get_confidence_score_dict(label_pipeline,
                                                                                        confidence_scores)

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
