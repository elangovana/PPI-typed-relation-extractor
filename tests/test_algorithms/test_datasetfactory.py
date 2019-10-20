from unittest import TestCase

from algorithms.dataset_factory import DatasetFactory
from datasets.custom_dataset_factory_base import CustomDatasetFactoryBase


class TestDatasetFactory(TestCase):

    def test_dataset_factory_names(self):
        # Arrange
        sut = DatasetFactory()
        expected_num_factories = 5

        # act
        class_names = sut.dataset_factory_names

        # assert
        self.assertEqual(len(class_names), expected_num_factories,
                         " The number of expected dataset factory classes doesnt match.. Check the number of classes that inhert from {}  and update the test if required !".format(
                             type(CustomDatasetFactoryBase)))

    def test_get_datasetfactory(self):
        # Arrange
        sut = DatasetFactory()
        class_names = sut.dataset_factory_names

        # Act
        obj = sut.get_datasetfactory(class_names[0])

        # assert
        self.assertIsInstance(obj, CustomDatasetFactoryBase)
