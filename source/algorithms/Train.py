import datetime
import logging
import os

import torch
import torch.utils.data
from torchtext.data import BucketIterator


class Train:

    def __init__(self):
        pass

    @property
    def logger(self):
        return logging.getLogger(__name__)

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
        :param data_iter: An iterable record of tuples of the form (label, features).  The each feature must be the index of word vocab
        :param model_network: A neural network
        :param loss_function: Pytorch loss function
        :param optimizer: Optimiser
        """
        losses = []
        best_val_acc = -1
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
            self.logger.info("Train set results:")
            self.print_confusion_matrix(actuals_train, predicted_train, output_dir, pos_label)

            # evaluate performance on validation set periodically
            if epoch % eval_every_n_epoch == 0:

                train_acc = 100. * n_correct / n_total

                val_acc, val_loss = self.calculate_val_loss(loss_function, model_network, val_iter, validation_iter,
                                                            output_dir, pos_label)

                self.logger.info(val_log_template.format((datetime.datetime.now() - start).seconds,
                                                         epoch, iterations, 1 + len(batch_x), len(train_iter),
                                                         100. * (1 + len(batch_x)) / len(train_iter), total_loss,
                                                         val_loss.item(), train_acc, val_acc))

                # update best valiation set accuracy
                if val_acc > best_val_acc:
                    self.save_snapshot(model_network, output_dir)

    def calculate_val_loss(self, loss_function, model_network, val_iter, validation_iter, output_dir, pos_label):
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
                actuals.extend(val_y)
                predicted.extend(pred_flat)
        self.logger.info("Validation set results:")
        self.print_confusion_matrix(actuals, predicted, output_dir, pos_label)
        val_acc = 100. * n_val_correct / len(validation_iter)
        return val_acc, val_loss

    def print_confusion_matrix(self, y_actual, y_pred, output_dir, pos_label):
        from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
        cnf_matrix = confusion_matrix(y_actual, y_pred)

        recall, precision, f1 = recall_score(y_actual, y_pred, pos_label=pos_label), precision_score(y_actual, y_pred,
                                                                                                     pos_label=pos_label), f1_score(
            y_actual, y_pred, pos_label=pos_label)

        filename = os.path.join(output_dir,
                                "predictedvsactual_{}".format(
                                    datetime.datetime.strftime(datetime.datetime.now(), format="%Y%m%d_%H%M%S")))
        self.save_data(y_pred, y_actual, filename)
        self.logger.info("Confusion matrix, full output in {}: \n{}".format(filename, cnf_matrix))

        self.logger.info("Precison {}, recall {}: f1 {}".format(precision, recall, f1))

    def save_snapshot(self, model, output_dir):
        # found a model with better validation set accuracy

        snapshot_prefix = os.path.join(output_dir, 'best_snapshot')
        snapshot_path = snapshot_prefix + 'model.pt'

        # save model, delete previous 'best_snapshot' files
        torch.save(model, snapshot_path)

    def save_data(self, pred, actual, outfile):
        # Write to output
        with open(outfile, "w") as out:
            for a, p in zip(actual, pred):
                out.write("{},{}\n".format(a, p))
