import datetime
import logging

import torch
import torch.utils.data
import torchtext
from torchtext.data import BucketIterator

from algorithms.ModelSnapshotCallback import ModelSnapshotCallback
from algorithms.result_scorer import ResultScorer, score_type_accuracy, score_type_precision
from algorithms.result_writer import ResultWriter


class Train:

    def __init__(self):
        self.snapshotter = None
        self.results_scorer = None
        self.results_writer = None

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
        self.__results_scorer__ = self.__results_scorer__ or ResultScorer()
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

    def __call__(self, data_iter, validation_iter, text_sort_key_lambda, model_network, loss_function, optimizer,
                 output_dir,
                 epoch=10, mini_batch_size=32,
                 eval_every_n_epoch=1, device_type="cpu", pos_label=1):
        """
    Runs train...
        :param validation_iter: Validation set
        :param epoch:
        :param mini_batch_size:
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
        train_iter, val_iter = BucketIterator.splits(
            (data_iter, validation_iter),  # we pass in the datasets we want the iterator to draw data from
            batch_size=mini_batch_size,
            device=torch.device(device_type),  # if you want to use the GPU, specify the GPU number here
            sort_key=text_sort_key_lambda,
            # the BucketIterator needs to be told what function it should use to group the data.
            sort_within_batch=True,
            repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
        )

        best_score = 0
        trainings_scores = []
        validation_scores = []
        score_measure = score_type_precision
        self.logger.info("using score : {}".format(score_measure))
        for epoch in range(epoch):
            train_iter.init_epoch()
            total_loss = 0
            n_correct, n_total = 0, 0
            actuals_train = []
            predicted_train = []
            for batch_x, batch_y in train_iter:
                iterations += 1
                # for feature, target in zip(batch_x, batch_y):

                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                model_network.train()
                model_network.zero_grad()

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                predicted = model_network(batch_x)

                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a tensor)
                loss = loss_function(predicted, torch.tensor(batch_y, dtype=torch.long))

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()

                losses.append(total_loss)

                actuals_train.extend(batch_y)
                predicted_train.extend(torch.max(predicted, 1)[1].view(batch_y.size()))

                n_correct += (torch.max(predicted, 1)[1].view(batch_y.size()) == batch_y).sum().item()
                n_total += len(batch_y)

            # Print training set confusion matrix
            self.logger.info("Train set result details:")
            self.results_writer(data_iter, actuals_train, predicted_train, pos_label, output_dir)
            train_results = self.results_scorer(y_actual=actuals_train, y_pred=predicted_train, pos_label=pos_label)
            trainings_scores.append({"epoch": epoch, "score": train_results})
            self.logger.info("Train set result details: {}".format(train_results))

            self.logger.info("Validation set result details:")
            val_actuals, val_predicted, val_loss = self.validate(loss_function, model_network, val_iter)
            self.results_writer(data_iter, val_actuals, val_predicted, pos_label, output_dir)
            val_results = self.results_scorer(y_actual=val_actuals, y_pred=val_predicted, pos_label=pos_label)
            validation_scores.append({"epoch": epoch, "score": val_results})
            # Print training set confusion matrix
            self.logger.info("Validation set result details: {} ".format(val_results))

            if val_results[score_measure] > best_score:
                best_results = (model_network, val_results, val_actuals, val_predicted)

                best_score = self.snapshotter(model_network, val_iter, best_score, output_dir=output_dir,
                                              pos_label=pos_label,
                                              metric=score_measure)

            # evaluate performance on validation set periodically
            self.logger.info(val_log_template.format((datetime.datetime.now() - start).seconds,
                                                     epoch, iterations, 1 + len(batch_x), len(train_iter),
                                                     100. * (1 + len(batch_x)) / len(train_iter), total_loss,
                                                     val_loss.item(), train_results[score_type_accuracy],
                                                     val_results[score_type_accuracy]))

        self.results_writer.dump_object(validation_scores, output_dir, "validation_scores_epoch")
        self.results_writer.dump_object(trainings_scores, output_dir, "training_scores_epoch")

        return best_results

    def validate(self, loss_function, model_network, val_iter):
        # switch model to evaluation mode
        model_network.eval()
        val_iter.init_epoch()
        # calculate accuracy on validation set
        n_val_correct, val_loss = 0, 0
        actuals = []
        predicted = []
        with torch.no_grad():
            for val_batch_idx, val_y in val_iter:
                pred_batch_y = model_network(val_batch_idx)

                pred_flat = torch.max(pred_batch_y, 1)[1].view(val_y.size())
                n_val_correct += (pred_flat == val_y).sum().item()
                val_loss = loss_function(pred_batch_y, val_y)
                actuals.extend(val_y.numpy().tolist())
                predicted.extend(pred_flat.numpy().tolist())

        return actuals, predicted, val_loss

    def predict(self, model_network, dataset):
        # switch model to evaluation mode
        model_network.eval()
        dataset_iterator = torchtext.data.Iterator(dataset, batch_size=1, train=False, sort=False, shuffle=False)
        predicted = []
        scores = []
        with torch.no_grad():
            for val_batch_idx, _ in dataset_iterator:
                pred_batch_y = model_network(val_batch_idx)
                scores.append(pred_batch_y)
                pred_binary = torch.max(pred_batch_y, 1)[1]
                pred_flat = pred_binary.view(pred_binary.size())

                predicted.extend(pred_flat.numpy().tolist())

        scores = [r.numpy().tolist() for r in torch.cat(scores, dim=0)]
        return predicted, scores
