from torch.utils.data import Dataset


class CustomDatasetBase(Dataset):

    @property
    def class_size(self):
        raise NotImplementedError

    @property
    def positive_label(self):
        raise NotImplementedError

    @property
    def feature_lens(self):
        raise NotImplementedError

    @property
    def entity_column_indices(self):
        raise NotImplementedError

    @property
    def text_column_index(self):
        raise NotImplementedError
