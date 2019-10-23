import logging

from modelnetworks.NetworkFactoryBase import NetworkFactoryBase
from modelnetworks.RelationExtractorSimpleResnetCnnPosNetwork import RelationExtractorSimpleResnetCnnPosNetwork


class RelationExtractorSimpleResnetCnnPosNetworkFactory(NetworkFactoryBase):

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self.logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

    def get_network(self, class_size, embedding_dim, feature_lens, **kwargs):
        dropout_rate_cnn = float(self._get_value(kwargs, "dropout_rate_cnn", ".5"))
        pooling_kernel_size = int(self._get_value(kwargs, "pooling_kernel_size", "3"))
        pool_stride = int(self._get_value(kwargs, "pool_stride", "2"))
        cnn_kernel_size = int(self._get_value(kwargs, "cnn_kernel_size", "3"))
        cnn_num_layers = int(self._get_value(kwargs, "cnn_num_layers", "3"))
        cnn_output = int(self._get_value(kwargs, "cnn_output", "64"))
        fc_layer_size = int(self._get_value(kwargs, "fc_layer_size", "64"))
        fc_drop_out_rate = float(self._get_value(kwargs, "fc_drop_out_rate", ".5"))
        input_drop_out_rate = float(self._get_value(kwargs, "input_drop_out_rate", ".8"))

        fine_tune_embeddings = bool(int(self._get_value(kwargs, "fine_tune_embeddings", "1")))

        model = RelationExtractorSimpleResnetCnnPosNetwork(class_size=class_size, embedding_dim=embedding_dim,
                                                           feature_lengths=feature_lens,
                                                           windows_size=cnn_kernel_size,
                                                           dropout_rate_cnn=dropout_rate_cnn,
                                                           cnn_output=cnn_output,
                                                           cnn_num_layers=cnn_num_layers,
                                                           cnn_stride=1, pool_kernel=pooling_kernel_size,
                                                           pool_stride=pool_stride, fc_layer_size=fc_layer_size,
                                                           fc_dropout_rate=fc_drop_out_rate,
                                                           input_dropout_rate=input_drop_out_rate,
                                                           fine_tune_embeddings=fine_tune_embeddings)

        return model
