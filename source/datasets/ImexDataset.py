import itertools

import pandas as pd


class ImexDataset:

    def __init__(self, interaction_type=None):
        self.interaction_type = interaction_type
        self.dataset = None
        self.labels = None

    def load(self, file_path):
        data_df = pd.read_json(file_path)

        # Filter based on interaction type if specific
        if self.interaction_type is not None:
            data_df = data_df.query('interactionType == "{}"'.format(self.interaction_type))

        # Extract labels
        labels = data_df[["isNegative"]]
        labels = pd.np.reshape(labels.values.tolist(), (-1,))

        # Prepare raw features
        data_df = data_df[["pubmedabstract", "interactionType", "participant1Alias", "participant2Alias"]]

        # Join alias...
        data_df['participant1Alias'] = data_df['participant1Alias'].map(
            lambda x: ", ".join(list(itertools.chain.from_iterable(x))))
        data_df['participant2Alias'] = data_df['participant2Alias'].map(
            lambda x: ", ".join(list(itertools.chain.from_iterable(x))))

        self.dataset, self.labels = data_df, labels

        # Return
        return self.dataset, self.labels
