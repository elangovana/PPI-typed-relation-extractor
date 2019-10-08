import numpy as np
import pandas as pd

from datasets.custom_dataset_base import CustomDatasetBase


class PpiAimedDataset(CustomDatasetBase):
    """
    Represents the custom PPI Aimed dataset
    """

    def __init__(self, file_path_or_dataframe, self_relations_filter=True, transformer=None):
        self._file_path = file_path_or_dataframe
        self.transformer = transformer
        # Read json
        if isinstance(file_path_or_dataframe, str):
            data_df = pd.read_json(self._file_path)
        elif isinstance(file_path_or_dataframe, pd.DataFrame):
            data_df = file_path_or_dataframe
        else:
            raise ValueError(
                "The type of argument file_path_or_dataframe  must be a str or pandas dataframe, but is {}".format(
                    type(file_path_or_dataframe)))

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

        # transform
        if self.transformer is not None:
            self.row_values = self.transformer(row_values)

        # remove the location offsets
        x = np.array(row_values)[[0, 1, 3]].tolist()

        # y
        y = self._labels[index].tolist()
        return x, y

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
    def entity_offset_indices(self):
        return [2, 4]

    @property
    def text_column_index(self):
        return 0
