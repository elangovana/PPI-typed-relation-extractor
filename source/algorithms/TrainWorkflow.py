import logging
import os

import pandas as pd
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

    def __call__(self, train_file, val_file, test_file=None):
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

        # Get binary average
        try:
            self.print_average_scores(val_actuals, val_predicted, pos_label=train.positive_label, average='binary')
        except ValueError as ev:
            self.logger.info("Could not be not be binary class {}, failed with error ".format(ev))

        self.print_average_scores(val_actuals, val_predicted, pos_label=train.positive_label, average='micro')

        self.print_average_scores(val_actuals, val_predicted, pos_label=train.positive_label, average='macro')

        tn, fp, fn, tp = confusion_matrix(val_actuals, val_predicted).ravel()

        self.logger.info("Confusion matrix: tn, fp, fn, tp  is {}".format((tn, fp, fn, tp)))

        if test_file is not None:
            train_pipeline = builder.get_trainpipeline()

            self.predict_test_set(dataset_factory, self.model_dir, self.out_dir, test_file, train_pipeline)

        return val_results, val_actuals, val_predicted

    def print_average_scores(self, val_actuals, val_predicted, pos_label=None, average='macro'):
        precision, recall, fscore, support = precision_recall_fscore_support(val_actuals, val_predicted,
                                                                             average=average,
                                                                             pos_label=pos_label)

        self.logger.info("{} average scores: precision, recall, fscore, support {}".format(
            average, (precision, recall, fscore, support)))

        self.logger.info("F-score {} : is {}".format(average, fscore))

    def predict_test_set(self, dataset_factory, model_dir, out_dir, test_file, train_pipeline):
        logger = logging.getLogger(__name__)
        logger.info("Evaluating test set {}".format(test_file))
        test_dataset = dataset_factory.get_dataset(test_file)
        predictor = train_pipeline.load(model_dir)
        predicted, confidence_scores = predictor(test_dataset)

        # Add results to raw dataset
        test_df = pd.read_json(test_file)
        logger.info("Test data shape: {}".format(test_df.shape))

        test_df["predicted"] = predicted
        test_df["confidence_scores"] = confidence_scores
        # # This is log softmax, convert to softmax prob
        # test_df["confidence_true"] = test_df.apply(lambda x: math.exp(x["confidence_scores"][True]), axis=1)
        # test_df["confidence_false"] = test_df.apply(lambda x: math.exp(x["confidence_scores"][False]), axis=1)
        predictions_file = os.path.join(out_dir, "predicted.json")
        test_df.to_json(predictions_file)

        logger.info("Evaluating test set complete, results in {}".format(predictions_file))
