import os
from unittest import TestCase

from algorithms.transform_berttext_tokenise import TransformBertTextTokenise


class ITTransformBertTextTokenise(TestCase):

    def test___init__(self):
        base_model_dir = os.path.join(os.path.dirname(__file__), "..", "temp", "biobert")

        assert len(os.listdir(
            base_model_dir)) >= 3, "The dir {} should contain the model bin and config and vocab files. If not download the biobert model".format(
            base_model_dir)

        # Arrange
        max_feature_lens = [30, 7, 7]
        case_insensitive = False
        sut = TransformBertTextTokenise(base_model_dir, max_feature_lens, case_insensitive)

        input = [
            # batch

            [
                # X
                [
                    # 3 columns
                    [
                        "This is a map PROTEIN1. PROTEIN1 phophorylates PROTEIN2"
                    ]
                    ,
                    [
                        "PROTEIN1"
                    ]
                    ,
                    [
                        "PROTEIN2"
                    ]
                ]
                ,
                # y
                [

                    "Yes"

                ]
            ]
        ]

        expected = [
            # batch
            [
                # x
                [

                    # 3 columns
                    [

                        ["[CLS]", 'This', 'is', 'a', 'map', 'PR', '##OT', '##EI', '##N', '##1', '.', 'PR', '##OT',
                         '##EI',
                         '##N', '##1', 'p', '##hop', '##hor', '##yla', '##tes', 'PR', '##OT', '##EI', '##N', '##2',
                         "[PAD]", "[PAD]", "[PAD]", "[SEP]"]

                    ]
                    ,
                    [
                        ["[CLS]", 'PR', '##OT', '##EI', '##N', '##1', "[SEP]"]
                    ]
                    ,
                    [
                        ["[CLS]", 'PR', '##OT', '##EI', '##N', '##2', "[SEP]"]
                    ]

                ]  # end of x

                # Y
                ,
                [
                    # batch size 1
                    "Yes"
                ]
            ]  # end of batch
        ]

        # Act
        actual = sut.fit_transform(input)
        print(actual)
        print(expected)


        # Assert
        self.assertSequenceEqual(expected, actual)
