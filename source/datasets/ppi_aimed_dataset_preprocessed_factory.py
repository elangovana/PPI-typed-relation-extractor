from datasets.PpiAimedDatasetPreprocessed import PpiAimedDatasetPreprocessed
from datasets.custom_dataset_factory_base import CustomDatasetFactoryBase


class PpiAimedDatasetPreprocessedFactory(CustomDatasetFactoryBase):

    def get_dataset(self, file_path):
        dataset = PpiAimedDatasetPreprocessed(file_path_or_dataframe=file_path)

        return dataset
