import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from preprocessor.Preprocessor import Preprocessor
from preprocessor.ProteinMasker import ProteinMasker

"""
Represents the custom PPIm dataset
"""


class PpiAimedDataset(Dataset):

    def __init__(self, file_path, self_relations_filter=True, transformer=None):
        self._file_path = file_path
        self.transformer = transformer or self._get_transformer()
        # Read json
        data_df = pd.read_json(self._file_path)

        # Filter interaction types if required
        if self_relations_filter:
            data_df = data_df.query('participant1 != participant2')

        # Filter features
        self._data_df = data_df[["passage", "participant1", "participant1_loc", "participant2", "participant2_loc"]]

        # Set up labels
        if "isValid" in data_df.columns:
            self._labels = data_df[["isValid"]]
            self._labels = np.reshape(self._labels.values.tolist(), (-1,))
        else:
            self._labels = np.reshape([-1] * data_df.shape[0], (-1,))

    def __len__(self):
        return self._data_df.shape[0]

    def __getitem__(self, index):
        row_values = self._data_df.iloc[index, :].tolist()
        # Convert to offset "22-40" 22
        row_values[2] = int(row_values[2].split("-")[0])
        row_values[4] = int(row_values[4].split("-")[0])

        transformed = np.array(self.transformer(row_values))[[0, 1, 3]].tolist()
        return transformed, self._labels[index].tolist()

    @property
    def class_size(self):
        return 2

    @property
    def positive_label(self):
        return True

    @property
    def feature_lens(self):
        return [150, 1, 1]

    @property
    def entity_column_indices(self):
        return [1, 3]

    @property
    def text_column_index(self):
        return 0

    def _get_transformer(self):
        mask = ProteinMasker(entity_column_indices=self.entity_column_indices, masks=["PROTEIN_1", "PROTEIN_2"],
                             text_column_index=self.text_column_index, entity_offset_indices=[2, 4])

        return Preprocessor([mask])
