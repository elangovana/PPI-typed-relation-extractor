import logging
import os
import tempfile
from unittest import TestCase

from algorithms.result_writer import ResultWriter


class TestResultsWriter(TestCase):

    def test___call__(self):
        # Arrange
        sut = ResultWriter()
        sut.logger.setLevel(logging.DEBUG)

        # Act
        out_dir = tempfile.mkdtemp()
        sut(x=[1, 2, 3], y_actual=[1, 0], y_pred=[1, 1], filename_prefix="test",
            output_dir=out_dir)

        # Assert

        self.assertGreater(len(os.listdir(out_dir)), 0, "The output directorymust contain atleast one file")

    def dump_object(self):
        # Arrange
        sut = ResultWriter()
        sut.logger.setLevel(logging.DEBUG)

        # Act
        out_dir = tempfile.mkdtemp()
        sut.dump_object([1, 2, 3], out_dir, filename_prefix="test")

        # Assert
        self.assertGreater(len(os.listdir(out_dir)), 0, "The output directorymust contain stleats one file")
