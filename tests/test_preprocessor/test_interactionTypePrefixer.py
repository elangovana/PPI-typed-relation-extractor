from unittest import TestCase

from preprocessor.InteractionTypePrefixer import InteractionTypePrefixer


class TestInteractionTypePrefixer(TestCase):
    def test_transform(self):
        # Arrange
        data = ["This is sample entity1 entity1", "entity1", "entity2", "phosphorylation"]

        expected = ["QUERYphosphorylation This is sample entity1 entity1", "entity1", "entity2", "phosphorylation"]

        sut = InteractionTypePrefixer(col_to_transform=0, prefixer_col_index=3)

        # Act
        actual = sut(data)

        # Assert
        self.assertSequenceEqual(expected, actual)
