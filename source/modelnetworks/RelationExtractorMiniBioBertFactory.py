import logging

from modelnetworks.BertNetworkFactoryBase import BertNetworkFactoryBase
from modelnetworks.RelationExtractorMiniBioBert import RelationExtractorMiniBioBert


class RelationExtractorMiniBioBertFactory(BertNetworkFactoryBase):

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

        num_layers = int(self._get_value(kwargs, "num_layers", "5"))

        model = RelationExtractorMiniBioBert(model_dir, class_size, num_layers=num_layers)

        return model
