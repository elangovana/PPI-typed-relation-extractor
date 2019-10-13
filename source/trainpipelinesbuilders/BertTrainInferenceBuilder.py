import logging

import numpy as np
from torch import nn

from algorithms.BertTrain import BertTrain
from algorithms.BertTrainInferencePipeline import BertTrainInferencePipeline
from algorithms.DataPipeline import DataPipeline
from algorithms.LabelPipeline import LabelPipeline
from algorithms.network_factory_locator import NetworkFactoryLocator
from algorithms.transform_berttext_token_to_index import TransformBertTextTokenToIndex
from algorithms.transform_berttext_tokenise import TransformBertTextTokenise
from algorithms.transform_label_encoder import TransformLabelEncoder
from algorithms.transform_label_rehaper import TransformLabelReshaper


class BertTrainInferenceBuilder:

    def __init__(self, dataset, model_dir, output_dir, epochs=100, patience_epochs=20,
                 extra_args=None, network_factory_name="RelationExtractorBioBertFactory"):
        self.network_factory_name = network_factory_name
        self.patience_epochs = patience_epochs
        self.model_dir = model_dir
        self.epochs = epochs
        self.dataset = dataset
        self.momentum = .9

        self.output_dir = output_dir
        self.additional_args = extra_args or {}

        self.batch_size = int(self._get_value(self.additional_args, "batchsize", "32"))

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self.logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def get_trainpipeline(self):
        # Text to token
        # 512 is the size of pretrained bio bert
        max_feature_lens = [min(l * 7, 512) for l in self.dataset.feature_lens]
        case_insensitive = False
        base_model_dir = self._get_value(self.additional_args, "pretrained_biobert_dir", None)
        assert base_model_dir is not None, "The value for base_model_dir must be passed and must be a valid dir with biobert artifacts"
        text_to_token = TransformBertTextTokenise(base_model_dir, max_feature_lens, case_insensitive)

        # Token to id
        token_to_index = TransformBertTextTokenToIndex(base_model_dir, case_insensitive,
                                                       text_col_index=self.dataset.text_column_index)

        data_pipeline = DataPipeline(text_to_index=None, preprocess_steps=None,
                                     processing_steps=[("token_Totext", text_to_token)
                                         , (("token_to_index", token_to_index))])

        # Label pipeline
        class_size = self.dataset.class_size
        label_reshaper = TransformLabelReshaper(num_classes=class_size)
        label_encoder = TransformLabelEncoder()
        label_pipeline = LabelPipeline(label_reshaper=label_reshaper, label_encoder=label_encoder)

        np_feature_lens = np.array(max_feature_lens)

        # network
        model_factory = NetworkFactoryLocator().get_factory(self.network_factory_name)
        model = model_factory.get_network(embedding_dim=None, class_size=class_size, feature_lens=np_feature_lens,
                                          **self.additional_args)

        self.logger.info("Using model {}".format(type(model)))
        self.logger.info("\n{}".format(model))

        # Loss function
        loss_function = nn.CrossEntropyLoss()
        self.logger.info("Using loss function {}".format(type(loss_function)))

        # Trainer
        trainer = BertTrain(epochs=self.epochs, early_stopping_patience=self.patience_epochs)

        pipeline = BertTrainInferencePipeline(model=model, loss_function=loss_function,
                                              trainer=trainer,
                                              model_dir=self.model_dir,
                                              batch_size=self.batch_size,
                                              label_pipeline=label_pipeline, data_pipeline=data_pipeline,
                                              class_size=class_size, pos_label=self.dataset.positive_label,
                                              output_dir=self.output_dir, additional_args=self.additional_args)

        return pipeline
