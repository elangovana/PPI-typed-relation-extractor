from datasets.custom_dataset_factory_base import CustomDatasetFactoryBase
from datasets.interaction_binary_dataset import InteractionBinaryDataset
from metrics.result_scorer_f1_binary_factory import ResultScorerF1BinaryFactory


class InteractionDatasetFactory(CustomDatasetFactoryBase):

    def get_metric_factory(self):
        return ResultScorerF1BinaryFactory()

    def get_dataset(self, file_path):
        dataset = InteractionBinaryDataset(file_path=file_path)

        return dataset
