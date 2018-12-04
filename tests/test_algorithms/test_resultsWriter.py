import tempfile
from unittest import TestCase

from algorithms.result_writer import ResultWriter


class TestResultsWriter(TestCase):

    def test___call__(self):
        # Arrange
        sut = ResultWriter()

        # Act
        sut(x=[1, 2, 3], y_actual=[1, 0], y_pred=[1, 1], pos_label=1, filename_prefix="test",
            output_dir=tempfile.mkdtemp())

        # Assert
