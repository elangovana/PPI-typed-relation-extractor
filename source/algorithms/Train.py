import glob
import os
import datetime

import torch
import torch.utils.data
from torchtext.data import Iterator, BucketIterator


class Train:

    def __init__(self):
        pass

    def __call__(self, data_iter, validation_iter, text_sort_key_lambda, model_network, loss_function, optimizer,
                 output_dir,
                 epoch=10, mini_batch_size=10,
                 eval_every_n_epoch=1, device=-1):
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
        n_correct, n_total, best_val_acc = 0, 0, -1
        start = datetime.datetime.now()
        iterations = 0
        val_log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))

        train_iter, val_iter = BucketIterator.splits(
            (data_iter, validation_iter),  # we pass in the datasets we want the iterator to draw data from
            batch_size=mini_batch_size,
            device=device,  # if you want to use the GPU, specify the GPU number here
            sort_key=text_sort_key_lambda,
            # the BucketIterator needs to be told what function it should use to group the data.
            sort_within_batch=True,
            repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
        )

        for epoch in range(epoch):
            train_iter.init_epoch()
            total_loss = 0
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

                n_correct += (torch.max(predicted, 1)[1].view(batch_y.size()) == batch_y).sum().item()
                n_total += len(batch_x)

                train_acc = 100. * n_correct / n_total

            # evaluate performance on validation set periodically
            if epoch % eval_every_n_epoch == 0:

                val_acc, val_loss = self.calculate_val_loss(loss_function, model_network, val_iter, validation_iter)

                print(val_log_template.format((datetime.datetime.now() - start).seconds,
                                              epoch, iterations, 1 + len(batch_x), len(train_iter),
                                              100. * (1 + len(batch_x)) / len(train_iter), loss.item(),
                                              val_loss.item(), train_acc, val_acc))

                # update best valiation set accuracy
                if val_acc > best_val_acc:
                    self.save_snapshot(model_network, output_dir)

            # End of each epoch
            print("{}\t{}".format(total_loss, train_acc))

    def calculate_val_loss(self, loss_function, model_network, val_iter, validation_iter):
        # switch model to evaluation mode
        model_network.eval()
        val_iter.init_epoch()
        # calculate accuracy on validation set
        n_val_correct, val_loss = 0, 0
        with torch.no_grad():
            for val_batch_idx, val_y in val_iter:
                answer = model_network(val_batch_idx)
                n_val_correct += (torch.max(answer, 1)[1].view(val_y.size()) == val_y).sum().item()
                val_loss = loss_function(answer, val_y)
        val_acc = 100. * n_val_correct / len(validation_iter)
        return val_acc, val_loss

    def save_snapshot(self, model, output_dir):
        # found a model with better validation set accuracy

        snapshot_prefix = os.path.join(output_dir, 'best_snapshot')
        snapshot_path = snapshot_prefix + 'model.pt'

        # save model, delete previous 'best_snapshot' files
        torch.save(model, snapshot_path)
