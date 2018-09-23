from dataextractors.BulkImexProteinInteractionsExtractor import BulkImexProteinInteractionsExtractor
from datatransformer.ImexDataTransformerAugmentAbstract import ImexDataTransformerAugmentAbstract


class SimplePipeline:

    def __init__(self, interactionlist=None):
        self.interactionlist = interactionlist or ['phosphorylation']
        self.data_reader = None
        self.pipeline_steps = []

    @property
    def data_reader(self):
        self.__bulkImexProteinInteractionsExtractor__ = self.__bulkImexProteinInteractionsExtractor__ or BulkImexProteinInteractionsExtractor(
            self.interactionlist)
        return self.__bulkImexProteinInteractionsExtractor__

    @data_reader.setter
    def data_reader(self, value):
        self.__bulkImexProteinInteractionsExtractor__ = value

    @property
    def pipeline_steps(self):
        return self.__pipeline_steps__

    @pipeline_steps.setter
    def pipeline_steps(self, value):
        self.__pipeline_steps__ = value

    def read_data(self, file_list, ):
        yield from self.data_reader.get_protein_interactions(file_list)

    def run(self, dataiter: iter):
        """
Runs a predefined set of steps in pipeline
        :param dataiter:  Iterable dataset
        """

        transformed_dataiter = dataiter
        for step in self.pipeline_steps:
            transformer = step[1]
            transformed_dataiter = transformer.transform(transformed_dataiter)

        return transformed_dataiter
