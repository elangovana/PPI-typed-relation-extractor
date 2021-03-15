from datasets.PpiMulticlassDataset import PpiMulticlassDataset
from datasets.custom_dataset_factory_base import CustomDatasetFactoryBase
from metrics.result_scorer_auc_macro_factory import ResultScorerAucMacroFactory
from preprocessor.Preprocessor import Preprocessor
from preprocessor.ProteinMasker import ProteinMasker


class PpiMulticlassDatasetFactory(CustomDatasetFactoryBase):

    def get_metric_factory(self):
        return ResultScorerAucMacroFactory()


    def get_dataset(self, file_path):
        dataset = PpiMulticlassDataset(file_path=file_path, interaction_type=None)

        mask = ProteinMasker(entity_column_indices=dataset.entity_column_indices, masks=dataset.entity_markers,
                             text_column_index=dataset.text_column_index)

        transformer = Preprocessor([mask])

        dataset.transformer = transformer

        return dataset
