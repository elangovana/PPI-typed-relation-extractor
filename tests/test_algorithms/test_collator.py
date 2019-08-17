from unittest import TestCase

from algorithms.Collator import Collator


class TestCollator(TestCase):

    def test__call__(self):
        # Arrange
        sut = Collator()
        data = [([
                     'The activated P60953 associated kinases (ACKs) are nonreceptor tyrosine kinases.',
                     'phosphorylation', 'Q07912', 'Q07912'], True)
            , ([
                   'The activated P60953 associated kinases (ACKs) are nonreceptor tyrosine kinases.',
                   'phosphorylation', 'Q07912', 'Q07912'], True)]

        expected = [[("The activated P60953 associated kinases (ACKs) are nonreceptor tyrosine kinases.",
                      "The activated P60953 associated kinases (ACKs) are nonreceptor tyrosine kinases."),
                     ("phosphorylation",
                      "phosphorylation"),
                     ("Q07912",
                      "Q07912"),
                     ("Q07912",
                      "Q07912")],
                    [True, True]]
        # Act
        actual = sut(data)
        print(actual)
        print(expected)
        # Assert
        self.assertSequenceEqual(expected, actual)
