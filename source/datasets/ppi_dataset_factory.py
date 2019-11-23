from datasets.PpiDataset import PPIDataset
from datasets.custom_dataset_factory_base import CustomDatasetFactoryBase
from metrics.result_scorer_f1_binary_factory import ResultScorerF1BinaryFactory
from preprocessor.InteractionTypePrefixer import InteractionTypePrefixer
from preprocessor.Preprocessor import Preprocessor
from preprocessor.ProteinMasker import ProteinMasker


class PpiDatasetFactory(CustomDatasetFactoryBase):

    def get_metric_factory(self):
        return ResultScorerF1BinaryFactory()


    def get_dataset(self, file_path):
        dataset = PPIDataset(file_path=file_path, interaction_type=None)

        mask = ProteinMasker(entity_column_indices=dataset.entity_column_indices, masks=["PROTEIN1", "PROTEIN2"],
                             text_column_index=dataset.text_column_index)

        interaction_type_prefixer = InteractionTypePrefixer(col_to_transform=dataset.text_column_index,
                                                            prefixer_col_index=3)
        transformer = Preprocessor([mask, interaction_type_prefixer])

        dataset.transformer = transformer

        return dataset
