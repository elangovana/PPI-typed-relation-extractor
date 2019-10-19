from datasets.PpiAimedDatasetYlhsieh import PpiAimedDatasetYlhsieh
from datasets.custom_dataset_factory_base import CustomDatasetFactoryBase


class PpiAimedDatasetFactoryYlhsieh(CustomDatasetFactoryBase):

    def get_dataset(self, file_path):
        dataset = PpiAimedDatasetYlhsieh(file_path_or_dataframe=file_path)

        return dataset
