import logging

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from algorithms.TrainInferenceBuilder import TrainInferenceBuilder
from algorithms.dataset_factory import DatasetFactory


class TrainWorkflow:

    def __init__(self, dataset_factory_name, network_factory_name, embedding_file, embedding_dim,
                 model_dir, out_dir,
                 epochs,
                 patience_epochs, extra_args):
        self.additionalargs = extra_args
        self.earlystoppingpatience = patience_epochs
        self.epochs = epochs
        self.out_dir = out_dir
        self.model_dir = model_dir
        self.embed_dim = embedding_dim
        self.embedding_file = embedding_file
        self.network_factory_name = network_factory_name
        self.dataset_factory_name = dataset_factory_name

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, train_file, val_file):
        dataset_factory = DatasetFactory().get_datasetfactory(self.dataset_factory_name)

        train, val = dataset_factory.get_dataset(train_file), dataset_factory.get_dataset(val_file)
        with open(self.embedding_file, "r") as embedding_handle:
            builder = TrainInferenceBuilder(dataset=train, embedding_dim=self.embed_dim,
                                            embedding_handle=embedding_handle,
                                            model_dir=self.model_dir, output_dir=self.out_dir, epochs=self.epochs,
                                            patience_epochs=self.earlystoppingpatience,
                                            extra_args=self.additionalargs,
                                            network_factory_name=self.network_factory_name)
            train_pipeline = builder.get_trainpipeline()
            train_pipeline(train, val)

            val_results, val_actuals, val_predicted = train_pipeline(train, val)
        precision, recall, fscore, support = precision_recall_fscore_support(val_actuals, val_predicted,
                                                                             average='binary',
                                                                             pos_label=train.positive_label)
        tn, fp, fn, tp = confusion_matrix(val_actuals, val_predicted).ravel()

        self.logger.info("Confusion matrix: tn, fp, fn, tp  is {}".format((tn, fp, fn, tp)))
        self.logger.info("Scores: precision, recall, fscore, support {}".format((precision, recall, fscore, support)))

        self.logger.info(" F-score is {}".format(fscore))

        return val_results, val_actuals, val_predicted
