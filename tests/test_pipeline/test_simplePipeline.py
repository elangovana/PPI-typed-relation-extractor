from unittest import TestCase
from unittest.mock import MagicMock

from pipeline.simplePipeline import SimplePipeline


class TestSimplePipeline(TestCase):

    def test_run(self):
        # Arrange
        # step 1
        mock_step1 = MagicMock()
        mock_step1.transform.return_value = [{"Itema": "1", "step": 1}]

        # step 2
        mock_step2 = MagicMock()
        mock_step2.transform.return_value = [{"Itema": "1", "step": 2}]

        sut = SimplePipeline()
        sut.pipeline_steps = [("mock_step1", mock_step1), ("mock_stpe2", mock_step2)]
        data = [{"Itema": "1"}]

        # Act
        actual = sut.run(dataiter=data)

        # Assert
        self.assertSequenceEqual(actual, [{"Itema": "1", "step": 2}])
