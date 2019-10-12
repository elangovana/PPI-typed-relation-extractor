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

        expected = [
            # batch
            [
                # x
                torch.tensor([[1188, 1110, 170, 4520, 11629, 14697, 27514, 2249, 1475, 119,
                               11629, 14697, 27514, 2249, 1475, 185, 23414, 13252, 22948, 3052,
                               11629, 14697, 27514, 2249, 1477, 0, 0, 0, 0, 0]])
                ,
                ['Yes']
            ]  # end of batch
        ]

        # Act
        actual = sut.fit_transform(input)

        # Assert
        # Entire n of batches
        self.assertEqual(len(actual), len(expected), "Number of batches do not match ")
        # Entire n of batches
        for (b_a_x, b_a_y), (b_e_x, b_e_y) in zip(actual, expected):
            self.assertIsInstance(b_a_x, torch.Tensor)

            # Tensor of columns
            self.assertSequenceEqual(b_a_x.shape, b_e_x.shape, "The tensor shape of actual and expected do not match")
            self.assertSequenceEqual(b_a_y, b_e_y, "The sequence of y  actual and expected do not match")
