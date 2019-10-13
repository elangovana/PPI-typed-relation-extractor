from trainpipelinesbuilders.BaseBuilderTrainFactory import BaseBuilderTrainFactory
from trainpipelinesbuilders.BertTrainInferenceBuilder import BertTrainInferenceBuilder


class BertBuilderTrainInferencePipelineFactory(BaseBuilderTrainFactory):

    def get_trainbuilder(self, dataset, model_dir, output_dir):
        return BertTrainInferenceBuilder(dataset, model_dir, output_dir)
