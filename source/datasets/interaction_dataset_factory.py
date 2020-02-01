from datasets.custom_dataset_factory_base import CustomDatasetFactoryBase
from datasets.interaction_dataset import InteractionDataset
from metrics.result_scorer_f1_macro_factory import ResultScorerF1MacroFactory


class InteractionDatasetFactory(CustomDatasetFactoryBase):

    def get_metric_factory(self):
        return ResultScorerF1MacroFactory()

    def get_dataset(self, file_path):
        dataset = InteractionDataset(file_path=file_path)

        return dataset
