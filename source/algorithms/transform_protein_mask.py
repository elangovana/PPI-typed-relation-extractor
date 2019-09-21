import logging


class TransformProteinMask:
    """
    Replaces the name of the protein in the abstract .
    """

    def __init__(self, entity_column_index, text_column_index, mask, entity_offset_index=None):
        """
Replaces the entity_column_index value in text_column_index with the mask.
If entity_offset is specified, then only the value in that location is replaced, else all mentions are replaced
        :type mask: str
        :type text_column_index: int
        :type entity_column_index: int
        :param entity_column_index: The zero_indexed col  of the entity to replace
        :param text_column_index: The zero_indexed col of the text to replace in
        :param mask: The mask string that is meant to replace the value in entity_column_index
        :param entity_offset_index: The zero_indexed col containing location position to replace within the textcolumn index
        """
        self.entity_offset_index = entity_offset_index
        self.mask = mask
        self.text_column_index = text_column_index
        self.entity_column_index = entity_column_index

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def fit(self, data_loader):
        pass

    def transform(self, x):
        self.logger.info("Running TransformProteinMask ")
        batches = []
        for idx, b in enumerate(x):
            b_x = b[0]
            b_y = b[1]

            text_column = b_x[self.text_column_index]
            entity_column = b_x[self.entity_column_index]
            offset = [-1 for _ in range(len(b_x))] if self.entity_offset_index is None else b_x[self.entity_offset_index]

            masked_text = []
            masked_entity = []
            for _, (t, e, offset_start) in enumerate(zip(text_column, entity_column, offset)):
                assert isinstance(e, str), "Entity column must be a string"
                assert isinstance(t, str), "text column must be a string"

                if offset_start == -1:
                    replaced_text = t.replace(e, self.mask)
                else:
                    offset_start = offset_start
                    offset_end = offset_start + len(e)
                    offset_text = t[offset_start: offset_end]
                    assert offset_text == e, "The text at offset_start {} must match entity {}, but found {}".format(
                        offset_start, e, offset_text)
                    replaced_text = t[:offset_start] + self.mask + t[offset_end:]

                masked_text.append(replaced_text)
                masked_entity.append(self.mask)

            transformed_b_x = b_x
            transformed_b_x[self.text_column_index] = masked_text
            transformed_b_x[self.entity_column_index] = masked_entity

            batches.append([transformed_b_x, b_y])
        self.logger.info("Completed TransformProteinMask ")

        return batches

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)
