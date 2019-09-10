import logging

import numpy as np
from torch import nn
from torch.optim import SGD, Adam

from algorithms.DataPipeline import DataPipeline
from algorithms.LabelPipeline import LabelPipeline
from algorithms.PretrainedEmbedderLoader import PretrainedEmbedderLoader
from algorithms.RelationExtractorBiLstmNetwork import RelationExtractorBiLstmNetwork
from algorithms.RelationExtractorCnnPosNetwork import RelationExtractorCnnPosNetwork
from algorithms.Train import Train
from algorithms.TrainInferencePipeline import TrainInferencePipeline
from algorithms.transform_label_encoder import TransformLabelEncoder
from algorithms.transform_label_rehaper import TransformLabelReshaper
from algorithms.transform_protein_mask import TransformProteinMask
from algorithms.transform_sentence_tokeniser import TransformSentenceTokenisor
from algorithms.transform_text_index import TransformTextToIndex


class TrainInferenceBuilder:

    def __init__(self, dataset, embedding_dim, embedding_handle, output_dir, epochs=100):
        self.epochs = epochs
        self.dataset = dataset
        self.learning_rate = .01
        self.momentum = .9
        self.embedding_handle = embedding_handle
        self.embedding_dim = embedding_dim
        self.output_dir = output_dir
        self.protein_mask = "PROTEIN_{}"

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def get_trainpipeline(self):
        # Embedder loader
        embedder_loader = PretrainedEmbedderLoader(TransformTextToIndex.pad_token)

        # preprocess steps TransformProteinMask
        preprocess_steps = []
        special_words = []
        for i in self.dataset.entity_column_indices:
            mask_format = self.protein_mask.format(i)
            transformer = TransformProteinMask(entity_column_index=i, text_column_index=self.dataset.text_column_index,
                                               mask=mask_format)
            special_words.append(mask_format)
            preprocess_steps.append(("mask_{}".format(i), transformer))

        # Add sentence tokenisor
        sentence_tokenisor = TransformSentenceTokenisor(text_column_index=self.dataset.text_column_index,
                                                        eos_token=TransformTextToIndex.eos_token())
        preprocess_steps.append(("Sentence_tokenisor", sentence_tokenisor))

        # Create data and label pipeline
        text_to_index = TransformTextToIndex(max_feature_lens=self.dataset.feature_lens, special_words=special_words)
        data_pipeline = DataPipeline(preprocess_steps=preprocess_steps, text_to_index=text_to_index)

        # Label pipeline
        class_size = self.dataset.class_size
        label_reshaper = TransformLabelReshaper(num_classes=class_size)
        label_encoder = TransformLabelEncoder()
        label_pipeline = LabelPipeline(label_reshaper=label_reshaper, label_encoder=label_encoder)

        np_feature_lens = np.array(self.dataset.feature_lens)

        # network
        model = RelationExtractorBiLstmNetwork(class_size=class_size, embedding_dim=self.embedding_dim,
                                               feature_lengths=np_feature_lens, hidden_size=150, dropout_rate_fc=0.5,
                                               kernal_size=4, fc_layer_size=30,
                                               lstm_dropout=.5)
        # model = RelationExtractorCnnPosNetwork(class_size=class_size, embedding_dim=self.embedding_dim,
        #                                        feature_lengths=np_feature_lens, cnn_output=250, dropout_rate_cnn=.5,
        #                                        dropout_rate_fc=0.5)
        self.logger.info("Using model {}".format(type(model)))

        # Optimiser
        # optimiser = SGD(lr=self.learning_rate, momentum=self.momentum, params=model.parameters())
        optimiser = Adam(params=model.parameters())
        self.logger.info("Using optimiser {}".format(type(optimiser)))

        # Loss function
        loss_function = nn.CrossEntropyLoss()
        self.logger.info("Using loss function {}".format(type(loss_function)))

        # Trainer
        trainer = Train()

        pipeline = TrainInferencePipeline(model=model, optimiser=optimiser, loss_function=loss_function,
                                          trainer=trainer, train_vocab_extractor=text_to_index,
                                          embedder_loader=embedder_loader,
                                          embedding_handle=self.embedding_handle, embedding_dim=self.embedding_dim,
                                          label_pipeline=label_pipeline, data_pipeline=data_pipeline,
                                          class_size=class_size, pos_label=self.dataset.positive_label,
                                          output_dir=self.output_dir, epochs=self.epochs)

        return pipeline
