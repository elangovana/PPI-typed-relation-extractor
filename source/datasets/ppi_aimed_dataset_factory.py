from datasets.PpiAimedDataset import PpiAimedDataset
from datasets.custom_dataset_factory_base import CustomDatasetFactoryBase
from preprocessor.Preprocessor import Preprocessor
from preprocessor.ProteinMasker import ProteinMasker


class PpiAimedDatasetFactory(CustomDatasetFactoryBase):

    def get_dataset(self, file_path):
        dataset = PpiAimedDataset(file_path_or_dataframe=file_path)

        mask = ProteinMasker(entity_column_indices=dataset.entity_column_indices, masks=["PROTEIN1", "PROTEIN2"],
                             text_column_index=dataset.text_column_index,
                             entity_offset_indices=dataset.entity_offset_indices)

        transformer = Preprocessor([mask])

        dataset.transformer = transformer

        return dataset
