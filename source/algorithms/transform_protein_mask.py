import logging

"""
Replaces the name of the protein in the abstract .
"""


class TransformProteinMask:

    def __init__(self, entity_column_index, text_column_index, mask):

        self.mask = mask
        self.text_column_index = text_column_index
        self.entity_column_index = entity_column_index

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, data_loader):
        pass

    def transform(self, x):

        batches = []
        for idx, b in enumerate(x):
            b_x = b[0]
            b_y = b[1]

            text_column = b_x[self.text_column_index]
            entity_column = b_x[self.entity_column_index]

            masked_text = []
            masked_entity = []
            for _, (t, e) in enumerate(zip(text_column, entity_column)):
                assert isinstance(e, str), "Entity column must be a string"
                assert isinstance(t, str), "text column must be a string"

                masked_text.append(t.replace(e, self.mask))
                masked_entity.append(self.mask)

            transformed_b_x = b_x
            transformed_b_x[self.text_column_index] = masked_text
            transformed_b_x[self.entity_column_index] = masked_entity

            batches.append([transformed_b_x, b_y])
        return batches

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)
