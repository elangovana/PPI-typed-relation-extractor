import datetime
import logging

import torch
import torch.utils.data

from algorithms.ModelSnapshotCallback import ModelSnapshotCallback
from algorithms.result_writer import ResultWriter
from metrics.result_scorer_f1_macro import ResultScorerF1Macro


class Train:

    def __init__(self, device=None, epochs=10, early_stopping_patience=20, results_scorer=None):

        self.early_stopping_patience = early_stopping_patience
        self.epochs = epochs
        self.snapshotter = None
        self.results_scorer = results_scorer
        self.results_writer = None
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def snapshotter(self):
        self.__snapshotter__ = self.__snapshotter__ or ModelSnapshotCallback()
        return self.__snapshotter__

    @snapshotter.setter
    def snapshotter(self, value):
        self.__snapshotter__ = value

    @snapshotter.setter
    def snapshotter(self, value):
        self.__snapshotter__ = value

    @property
    def results_scorer(self):
        self.__results_scorer__ = self.__results_scorer__ or ResultScorerF1Macro()
        return self.__results_scorer__

    @results_scorer.setter
    def results_scorer(self, value):
        self.__results_scorer__ = value

    @property
    def results_writer(self):
        self.__results_writer__ = self.__results_writer__ or ResultWriter()
        return self.__results_writer__

    @results_writer.setter
    def results_writer(self, value):
        self.__results_writer__ = value

    def __call__(self, data_iter, validation_iter, model_network, loss_function, optimizer, model_dir,
                 output_dir,

                 eval_every_n_epoch=1, pos_label=1):
        """
    Runs train...
        :param validation_iter: Validation set
        :param epochs:
        :param device: For CPU -1, else set GPU device id
        :type eval_every_n_epoch: int
        :param data_iter: Torchtext dataset object. The each feature must be the index of word vocab
        :param model_network: A neural network
        :param loss_function: Pytorch loss function
        :param optimizer: Optimiser
        """
        losses = []
        best_results = None
        start = datetime.datetime.now()
        iterations = 0
        val_log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
        val_log_template = "Run {}".format(val_log_template)

        lowest_loss = None
        best_score = None
        trainings_scores = []
        validation_scores = []
        no_improvement_epochs = 0
        self.logger.info("using score : {}".format(type(self.results_scorer)))
        model_network.to(device=self.device)
        for epoch in range(self.epochs):

            self.logger.debug("Running epoch {}".format(self.epochs))

            for idx, batch in enumerate(data_iter):
                self.logger.debug("Running batch {}".format(idx))
                batch_x = [t.to(device=self.device) for t in batch[0]]
                batch_y = batch[1].to(device=self.device)
                # batch_x = torch.Tensor(batch_x)
                iterations += 1

                # Step 2. train
                model_network.train()
                model_network.zero_grad()

                # Step 3. Run the forward pass
                # words
                self.logger.debug("Running forward")
                predicted = model_network(batch_x)

                # Step 4. Compute loss
                self.logger.debug("Running loss")
                loss = loss_function(predicted, batch_y)

                # Step 5. Do the backward pass and update the gradient
                self.logger.debug("Running backwoard")
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            actuals_train, predicted_train, train_loss = self.validate(loss_function, model_network, data_iter)

            # Print training set confusion matrix
            self.logger.info("Train set result details:")
            self.results_writer(data_iter, actuals_train, predicted_train, output_dir)
            train_score = self.results_scorer(y_actual=actuals_train, y_pred=predicted_train,
                                              pos_label=pos_label.item())
            trainings_scores.append({"epoch": epoch, "score": train_score, "loss": train_loss})
            self.logger.info("Train set result details: {}".format(train_score))

            self.logger.info("Validation set result details:")
            val_actuals, val_predicted, val_loss = self.validate(loss_function, model_network, validation_iter)
            self.results_writer(validation_iter, val_actuals, val_predicted, output_dir)
            val_score = self.results_scorer(y_actual=val_actuals, y_pred=val_predicted, pos_label=pos_label.item())
            validation_scores.append({"epoch": epoch, "score": val_score, "loss": val_loss})
            # Print training set confusion matrix
            self.logger.info("Validation set result details: {} ".format(val_score))

            # Snapshot best score
            if (best_score is None or val_score > best_score):

                best_results = (val_score, val_actuals, val_predicted)
                self.logger.info(
                    "Snapshotting because the current score {} is greater than {} ".format(val_score, best_score))
                self.snapshotter(model_network, output_dir=model_dir)

                best_score = val_score
                lowest_loss = val_loss
                no_improvement_epochs = 0

            # Here is the score if the same, but lower loss
            elif best_score == val_score and (lowest_loss is None or val_loss < lowest_loss):
                best_results = (val_score, val_actuals, val_predicted)

                self.logger.info(
                    "Snapshotting because the current loss {} is lower than {} ".format(val_loss, lowest_loss))
                self.snapshotter(model_network, output_dir=model_dir)

                lowest_loss = val_loss
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            # evaluate performance on validation set periodically
            self.logger.info(val_log_template.format((datetime.datetime.now() - start).seconds,
                                                     epoch, iterations, 1 + len(batch_x), len(data_iter),
                                                     100. * (1 + len(batch_x)) / len(data_iter), train_loss,
                                                     val_loss, train_score,
                                                     val_score))

            print("###score: train_loss### {}".format(train_loss))
            print("###score: val_loss### {}".format(val_loss))
            print("###score: train_fscore### {}".format(train_score))
            print("###score: val_fscore### {}".format(val_score))

            if no_improvement_epochs > self.early_stopping_patience:
                self.logger.info("Early stopping.. with no improvement in {}".format(no_improvement_epochs))
                break

        self.results_writer.dump_object(validation_scores, output_dir, "validation_scores_epoch")
        self.results_writer.dump_object(trainings_scores, output_dir, "training_scores_epoch")

        return best_results

    def validate(self, loss_function, model_network, val_iter):
        # switch model to evaluation mode
        model_network.eval()
        # calculate accuracy on validation set
        n_val_correct, val_loss = 0, 0

        scores = []
        with torch.no_grad():
            actuals = torch.tensor([]).to(device=self.device)
            predicted = torch.tensor([]).to(device=self.device)
            for idx, val in enumerate(val_iter):
                val_batch_idx = [t.to(device=self.device) for t in val[0]]
                val_y = val[1].to(device=self.device)
                pred_batch_y = model_network(val_batch_idx)
                scores.append([pred_batch_y])
                pred_flat = torch.max(pred_batch_y, 1)[1].view(val_y.size())
                n_val_correct += (pred_flat == val_y).sum().item()
                val_loss += loss_function(pred_batch_y, val_y).item()
                actuals = torch.cat([actuals.long(), val_y])
                predicted = torch.cat([predicted.long(), pred_flat])

        self.logger.debug("The validation confidence scores are {}".format(scores))
        return actuals.cpu().numpy().tolist(), predicted.cpu().numpy().tolist(), val_loss
