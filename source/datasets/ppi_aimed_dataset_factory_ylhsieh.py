from datasets.PpiAimedDatasetYlhsieh import PpiAimedDatasetYlhsieh
from datasets.custom_dataset_factory_base import CustomDatasetFactoryBase
from metrics.result_scorer_f1_binary_factory import ResultScorerF1BinaryFactory


class PpiAimedDatasetFactoryYlhsieh(CustomDatasetFactoryBase):

    def get_metric_factory(self):
        return ResultScorerF1BinaryFactory()


    def get_dataset(self, file_path):
        dataset = PpiAimedDatasetYlhsieh(file_path_or_dataframe=file_path)

        return dataset
