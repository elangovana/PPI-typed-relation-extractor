class BaseBuilderTrainFactory:

    def get_trainbuilder(self, dataset, model_dir, output_dir):
        raise NotImplementedError
