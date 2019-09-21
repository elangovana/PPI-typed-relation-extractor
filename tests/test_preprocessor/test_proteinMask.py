from unittest import TestCase

from preprocessor.ProteinMasker import ProteinMasker


class TestProteinMasker(TestCase):
    def test_transform(self):
        # Arrange
        data = ["This is sample entity1 entity1", "entity1", "entity2", "phosphorylation"]

        expected = ["This is sample PROTEIN_1 PROTEIN_1", "PROTEIN_1", "entity2", "phosphorylation"]

        sut = ProteinMasker(entity_column_indices=[1], text_column_index=0, masks=["PROTEIN_1"])

        # Act
        actual = sut(["This is sample entity1 entity1", "entity1", "entity2", "phosphorylation"])

        # Assert
        self.assertSequenceEqual(expected, actual)

    def test_transform_with_offset(self):
        # Arrange

        # Mock data set
        data = ["entity2 This is sample entity1 entity1 ", "entity1", 23, "entity2", 0, "phosphorylation"]

        expected = ["entity2 This is sample PROTEIN_1 entity1 ", "PROTEIN_1", 23, "entity2", 0, "phosphorylation"]

        sut = ProteinMasker(entity_column_indices=[1], text_column_index=0, masks=["PROTEIN_1"],
                            entity_offset_indices=[2])

        # Act
        actual = sut(data)

        # Assert
        self.assertSequenceEqual(expected, actual)

    def test_transform_with_multi_offset(self):
        # Arrange
        data = ["entity2 This is sample entity1 entity2 entity2", "entity2", 31, "entity1", 23, "phosphorylation"]

        expected = ["entity2 This is sample PROTEIN_1 PROTEIN_2 entity2", "PROTEIN_2", 31, "PROTEIN_1", 23,  "phosphorylation"]

        sut = ProteinMasker(entity_column_indices=[1, 3], text_column_index=0, masks=["PROTEIN_2", "PROTEIN_1"],
                            entity_offset_indices=[2, 4])

        # Act
        actual = sut(data)

        # Assert
        self.assertSequenceEqual(expected, actual)
