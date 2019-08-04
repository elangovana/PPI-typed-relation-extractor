import numpy as np
from torch import nn
from torch.optim import SGD

from algorithms.DataPipeline import DataPipeline
from algorithms.LabelPipeline import LabelPipeline
from algorithms.PretrainedEmbedderLoader import PretrainedEmbedderLoader
from algorithms.RelationExtractorCnnPosNetwork import RelationExtractorCnnPosNetwork
from algorithms.Train import Train
from algorithms.TrainInferencePipeline import TrainInferencePipeline


class CnnPosTrainInferenceFactory:

    def __init__(self, dataset, embedding_handle, output_dir):
        self.dataset = dataset
        self.learning_rate = .001
        self.momentum = .9
        self.embedding_handle = embedding_handle
        self.embedding_dim = 200
        self.output_dir = output_dir

    def get_trainpipeline(self):
        # Embedder loader
        embedder_loader = PretrainedEmbedderLoader()

        # Create data and label pipeline
        data_pipeline = DataPipeline(max_feature_lens=self.dataset.feature_lens
                                     , embeddings_handle=self.embedding_handle,
                                     pretrained_embedder_loader=embedder_loader)

        label_pipeline = LabelPipeline()

        np_feature_lens = np.array(self.dataset.feature_lens)
        model = RelationExtractorCnnPosNetwork(class_size=self.dataset, embedding_dim=self.embedding_dim,
                                               feature_lengths=np_feature_lens)

        # Optimiser
        optimiser = SGD(lr=self.learning_rate, momentum=self.momentum, params=model.parameters())

        # Loss function
        loss_function = nn.CrossEntropyLoss()

        # Trainer
        trainer = Train()

        pipeline = TrainInferencePipeline(model=model, optimiser=optimiser,
                                          loss_function=loss_function,
                                          trainer=trainer,
                                          data_pipeline=data_pipeline,
                                          label_pipeline=label_pipeline,
                                          class_size=self.dataset.class_size,
                                          embedding_dim=self.embedding_dim,
                                          embedder_loader=embedder_loader,
                                          embedding_handle=self.embedding_handle,
                                          output_dir=self.output_dir)

        return pipeline
