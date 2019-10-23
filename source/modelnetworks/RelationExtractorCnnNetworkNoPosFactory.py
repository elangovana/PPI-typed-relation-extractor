import logging

from modelnetworks.NetworkFactoryBase import NetworkFactoryBase
from modelnetworks.RelationExtractorCnnNetworkNoPos import RelationExtractorCnnNetworkNoPos


class RelationExtractorCnnNetworkNoPosFactory(NetworkFactoryBase):

    def get_network(self, class_size, embedding_dim, feature_lens, **kwargs):
        dropout_rate_cnn = float(self._get_value(kwargs, "dropout_rate_cnn", ".5"))
        cnn_output = int(self._get_value(kwargs, "cnn_output", "100"))
        fc_drop_out_rate = float(self._get_value(kwargs, "fc_drop_out_rate", ".5"))

        fine_tune_embeddings = bool(int(self._get_value(kwargs, "fine_tune_embeddings", "1")))

        model = RelationExtractorCnnNetworkNoPos(class_size=class_size, embedding_dim=embedding_dim,
                                                 feature_lengths=feature_lens, cnn_output=cnn_output,
                                                 dropout_rate_cnn=dropout_rate_cnn,
                                                 dropout_rate_fc=fc_drop_out_rate,
                                                 fine_tune_embeddings=fine_tune_embeddings)

        return model

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self.logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value
