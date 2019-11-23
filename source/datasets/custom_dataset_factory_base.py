class CustomDatasetFactoryBase:

    def get_dataset(self, file_path):
        raise NotImplementedError

    def get_metric_factory(self):
        raise NotImplementedError
