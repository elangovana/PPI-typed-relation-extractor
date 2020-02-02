import os
from unittest import TestCase

from modelnetworks.RelationExtractorMiniBioBert import RelationExtractorMiniBioBert


class ITRelationExtractorMiniBioBert(TestCase):
    def test___init__(self):
        base_model_dir = os.path.join(os.path.dirname(__file__), "..", "temp", "biobert")

        assert len(os.listdir(
            base_model_dir)) >= 2, "The dir {} should contain the model bin and config files. If not download the biobert model".format(
            base_model_dir)

        # Act
        sut = RelationExtractorMiniBioBert(base_model_dir, 2)
