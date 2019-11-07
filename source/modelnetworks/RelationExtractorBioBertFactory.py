import logging

from modelnetworks.BertNetworkFactoryBase import BertNetworkFactoryBase
from modelnetworks.RelationExtractorBioBert import RelationExtractorBioBert


class RelationExtractorBioBertFactory(BertNetworkFactoryBase):

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self.logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

    def get_network(self, class_size, embedding_dim, feature_lens, **kwargs):
        model_dir = self._get_value(kwargs, "pretrained_biobert_dir", None)

        assert model_dir is not None, "The model directory is mandatory and must contain the pretrained Biobert artifacts"

        model = RelationExtractorBioBert(model_dir, class_size)

        return model
