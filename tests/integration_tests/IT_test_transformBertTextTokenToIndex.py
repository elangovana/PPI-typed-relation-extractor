import os
from unittest import TestCase

import torch

from algorithms.transform_berttext_token_to_index import TransformBertTextTokenToIndex


class ITTransformBertTextTokenToIndex(TestCase):

    def test__call__(self):
        base_model_dir = os.path.join(os.path.dirname(__file__), "..", "temp", "biobert")

        assert len(os.listdir(
            base_model_dir)) >= 3, "The dir {} should contain the model bin and config and vocab files. If not download the biobert model".format(
            base_model_dir)

        # Arrange
        text_index = 0
        case_insensitive = False
        sut = TransformBertTextTokenToIndex(base_model_dir, case_insensitive, text_col_index=text_index)

        input = [
            # batch
            [
                # x
                [

                    # 3 columns
                    [

                        ['This', 'is', 'a', 'map', 'PR', '##OT', '##EI', '##N', '##1', '.', 'PR', '##OT', '##EI',
                         '##N', '##1', 'p', '##hop', '##hor', '##yla', '##tes', 'PR', '##OT', '##EI', '##N', '##2',
                         "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"]

                    ]
                    ,
                    [
                        ['PR', '##OT', '##EI', '##N', '##1']
                    ]
                    ,
                    [
                        ['PR', '##OT', '##EI', '##N', '##2']
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

        # Assert
        # Entire len
        self.assertEqual(len(actual), len(input))
        # Entire n of batches
        self.assertEqual(len(actual[0]), len(input[0]))
        # Size of columns
        self.assertEqual(len(actual[0][0]), 1)
        # Tensor of columns
        self.assertEqual(len(actual[0][0][text_index]), len(input[0][0][text_index]))
        self.assertIsInstance(actual[0][0][text_index], torch.Tensor)
