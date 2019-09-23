import logging

import numpy


class ProteinMasker:
    """
    Replaces the name of the protein in the abstract .
    """

    def __init__(self, entity_column_indices, text_column_index, masks, entity_offset_indices=None):
        """
Replaces the entity_column_index value in text_column_index with the mask.
If entity_offset is specified, then only the value in that location is replaced, else all mentions are replaced
        :type mask: str
        :type text_column_index: int
        :type entity_column_indices: List
        :param entity_column_indices: The zero_indexed col  of the entity to replace
        :param text_column_index: The zero_indexed col of the text to replace in
        :param mask: The list of mask string that is meant to replace the value in entity_column_index
        :param entity_offset_indices: The zero_indexed col containing location position to replace within the textcolumn index
        """
        assert len(entity_column_indices) == len(
            masks), "The length {} of entity_columns muct match the length of masks {}".format(
            len(entity_column_indices), len(masks))

        if entity_offset_indices is not None:
            assert len(entity_column_indices) == len(
                entity_offset_indices), "The length {} of entity_columns muct match the length of entity_offset_indices {}".format(
                len(entity_column_indices), len(entity_offset_indices))

        self.entity_offset_indices = entity_offset_indices or None
        self.masks = masks
        self.text_column_index = text_column_index
        self.entity_column_indices = entity_column_indices

        # # Sort by
        # if self.entity_offset_indices is not None:
        #     sorted_index = numpy.argsort(self.entity_offset_indices)
        #     self.entity_offset_indices = numpy.array(self.entity_offset_indices)[sorted_index].tolist()
        #     self.masks = numpy.array(self.masks)[sorted_index].tolist()
        #     self.text_column_index = text_column_index
        #     self.entity_column_indices = numpy.array(self.entity_column_indices)[sorted_index].tolist()

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, row_x):
        if self.entity_offset_indices is not None:
            offset_values = numpy.array(row_x)[self.entity_offset_indices].astype(int).tolist()
            sorted_c_index = numpy.argsort(offset_values)

            sorted_entities_cols = numpy.array(self.entity_column_indices)[sorted_c_index].tolist()
            sorted_offset_cols = numpy.array(self.entity_offset_indices)[sorted_c_index].tolist()
            sorted_masks = numpy.array(self.masks)[sorted_c_index].tolist()
        else:
            sorted_entities_cols = self.entity_column_indices
            sorted_offset_cols = [None for _ in range(len(sorted_entities_cols))]
            sorted_masks = self.masks

        text = row_x[self.text_column_index]
        adj = 0
        for ei, oi, m in zip(sorted_entities_cols, sorted_offset_cols, sorted_masks):
            e = row_x[ei]

            if oi == None:
                text = text.replace(e, m)
            else:
                pos_s = row_x[oi] + adj
                pos_e = pos_s + len(e)
                offset_text = text[pos_s: pos_e]
                # assert offset_text == e, "The text at offset_start {} must match entity '{}', but found '{}'".format(
                #     pos_s, ei, offset_text)
                if offset_text != e:
                    self.logger.warning(
                        "The text at offset_start {} must match entity '{}', but found '{}' for text \n{}".format(pos_s, ei,
                                                                                                    offset_text, text))
                text = text[:pos_s] + m + text[pos_e:]
                adj = len(m) - len(row_x[ei])
            row_x[ei] = m

        row_x[self.text_column_index] = text

        return row_x

    # def transform(self, x):
    #     self.logger.info("Running TransformProteinMask ")
    #     batches = []
    #     for idx, b in enumerate(x):
    #         b_x = b[0]
    #         b_y = b[1]
    #
    #         transformed_b_x = b_x
    #
    #         # Loop through each column
    #         for ci, entity_column_index in enumerate(self.entity_column_indices):
    #             entity_column = transformed_b_x[entity_column_index]
    #             text_column = transformed_b_x[self.text_column_index]
    #
    #             entity_offset_index = None
    #             if self.entity_offset_indices is not None:
    #                 entity_offset_index = self.entity_offset_indices[ci]
    #
    #             offset = [-1 for _ in range(len(transformed_b_x))] if entity_offset_index is None else transformed_b_x[
    #                 entity_offset_index]
    #
    #             masked_text = []
    #             masked_entity = []
    #
    #             adjustments = [0 for _ in range(len(transformed_b_x))]
    #
    #             mask = self.masks[ci]
    #
    #             # loop through each row
    #             for _, (t, e, offset_start, adj) in enumerate(zip(text_column, entity_column, offset, adjustments)):
    #                 assert isinstance(e, str), "Entity column must be a string"
    #                 assert isinstance(t, str), "text column must be a string"
    #
    #                 if offset_start == -1:
    #                     replaced_text = t.replace(e, mask)
    #                     adjustments.append([0])
    #                 else:
    #                     offset_start = offset_start + adj
    #                     offset_end = offset_start + len(e)
    #                     offset_text = t[offset_start: offset_end]
    #                     assert offset_text == e, "The text at offset_start {} must match entity {}, but found '{}'".format(
    #                         offset_start, e, offset_text)
    #                     replaced_text = t[:offset_start] + mask + t[offset_end:]
    #                     adjustments.append(len(e) - len(mask))
    #                 masked_text.append(replaced_text)
    #                 masked_entity.append(mask)
    #
    #             transformed_b_x[self.text_column_index ] = masked_text
    #             transformed_b_x[entity_column_index] = masked_entity
    #
    #         batches.append([transformed_b_x, b_y])
    #     self.logger.info("Completed TransformProteinMask ")
    #
    #     return batches

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)
