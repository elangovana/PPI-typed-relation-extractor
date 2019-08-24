import numpy as np
import pandas as pd
from torch.utils.data import Dataset

"""
Represents the custom PPI dataset
"""


class PPIDataset(Dataset):

    def __init__(self, file_path, interaction_type=None):
        self._file_path = file_path
        # Read json
        data_df = pd.read_json(self._file_path)

        # Filter interaction types if required
        if interaction_type is not None:
            data_df = data_df.query('interactionType == "{}"'.format(interaction_type))

        # Filter features
        self._data_df = data_df[["normalised_abstract", "interactionType", "participant1Id", "participant2Id"]]

        # Set up labels
        if "isValid" in data_df.columns:
            self._labels = data_df[["isValid"]]
            self._labels = np.reshape(self._labels.values.tolist(), (-1,))
        else:
            self._labels = np.reshape([-1] * data_df.shape[0], (-1,))

    def __len__(self):
        return self._data_df.shape[0]

    def __getitem__(self, index):
        return self._data_df.iloc[index, :].tolist(), self._labels[index].tolist()

    @property
    def class_size(self):
        return 2

    @property
    def positive_label(self):
        return True

    @property
    def feature_lens(self):
        return [250, 1, 1, 1]

    @property
    def entity_column_indices(self):
        return [1, 2]

    @property
    def text_column_index(self):
        return 0
