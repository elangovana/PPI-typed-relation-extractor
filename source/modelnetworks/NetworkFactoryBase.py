class NetworkFactoryBase:

    def get_network(self, class_size, embedding_dim, feature_lens, **kwargs):
        raise NotImplementedError
