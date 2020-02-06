# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
import os
import tempfile
from io import StringIO
from unittest import TestCase

from algorithms.InferencePipeline import InferencePipeline
from algorithms.TrainInferenceBuilder import TrainInferenceBuilder
from algorithms.dataset_factory import DatasetFactory


class TestSitInferencePipeline(TestCase):

    def test_should_run(self):
        # Arrange

        base_path = tempfile.mkdtemp()
        out_dir = os.path.join(base_path, "model_artifacts")
        os.mkdir(out_dir)

        # Run training
        mock_dataset_train, scorer = self._get_ppidataset()
        mock_dataset_val, _ = self._get_ppidataset()
        train_pipeline = self._get_sut_train_pipeline(mock_dataset_train, out_dir=out_dir, epochs=20, scorer=scorer)
        train_pipeline(mock_dataset_train, mock_dataset_val)

        sample_file = self._get_train_file()
        dataset, _ = self._get_ppidataset()

        sut = InferencePipeline()

        # Act + assert
        sut.run(artifactsdir=base_path, data_file=sample_file, dataset=dataset, out_dir=base_path)

    def _get_sut_train_pipeline(self, mock_dataset, out_dir=tempfile.mkdtemp(), epochs=5, scorer=None):
        embedding = StringIO(
            "\n".join(["4 3", "hat 0.2 .34 0.8", "mat 0.5 .34 0.8", "entity1 0.5 .55 0.8", "entity2 0.3 .55 0.9"]))
        factory = TrainInferenceBuilder(dataset=mock_dataset, embedding_handle=embedding, embedding_dim=3,
                                        output_dir=out_dir, model_dir=out_dir, epochs=epochs, results_scorer=scorer)
        sut = factory.get_trainpipeline()
        return sut

    def _get_ppidataset(self):
        # Arrange
        train_file = self._get_train_file()
        factory = self._get_dataset()
        dataset = factory.get_dataset(train_file)
        scorer = factory.get_metric_factory().get()

        return dataset, scorer

    def _get_dataset(self):
        factory = DatasetFactory().get_datasetfactory("PpiDatasetFactory")
        return factory

    def _get_train_file(self):
        train_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_train.json")
        return train_file
