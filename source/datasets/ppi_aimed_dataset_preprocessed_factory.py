from datasets.PpiAimedDatasetPreprocessed import PpiAimedDatasetPreprocessed
from datasets.custom_dataset_factory_base import CustomDatasetFactoryBase
from metrics.result_scorer_f1_binary_factory import ResultScorerF1BinaryFactory


class PpiAimedDatasetPreprocessedFactory(CustomDatasetFactoryBase):

    def get_metric_factory(self):
        return ResultScorerF1BinaryFactory()


    def get_dataset(self, file_path):
        dataset = PpiAimedDatasetPreprocessed(file_path_or_dataframe=file_path)

        return dataset
