import torch
import torch.utils.data
from torchtext.data import Iterator, BucketIterator


class Train:

    def __init__(self):
        pass

    def __call__(self, data_iter, validation_iter, text_sort_key_lambda, model_network, loss_function, optimizer,
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
        n_correct, n_total = 0, 0

        train_iter, val_iter = BucketIterator.splits(
            (data_iter, validation_iter),  # we pass in the datasets we want the iterator to draw data from
            batch_size=mini_batch_size,
            device=device,  # if you want to use the GPU, specify the GPU number here
            sort_key=text_sort_key_lambda,
            # the BucketIterator needs to be told what function it should use to group the data.
            sort_within_batch=False,
            repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
        )

        for epoch in range(epoch):
            total_loss = 0
            for batch_x, batch_y in train_iter:
                # for feature, target in zip(batch_x, batch_y):

                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
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

            # # evaluate performance on validation set periodically
            # if epoch % eval_every_n_epoch == 0:
            #
            #     # switch model to evaluation mode
            #     model_network.eval()
            #     dev_iter.init_epoch()
            #
            #     # calculate accuracy on validation set
            #     n_dev_correct, dev_loss = 0, 0
            #     with torch.no_grad():
            #         for dev_batch_idx, dev_batch in enumerate(dev_iter):
            #             answer = model(dev_batch)
            #             n_dev_correct += (torch.max(answer, 1)[1].view(
            #                 dev_batch.label.size()) == dev_batch.label).sum().item()
            #             dev_loss = criterion(answer, dev_batch.label)
            #     dev_acc = 100. * n_dev_correct / len(dev)
            #
            #     print(dev_log_template.format(time.time() - start,
            #                                   epoch, iterations, 1 + batch_idx, len(train_iter),
            #                                   100. * (1 + batch_idx) / len(train_iter), loss.item(),
            #                                   dev_loss.item(), train_acc, dev_acc))
            #
            #     # update best valiation set accuracy
            #     if dev_acc > best_dev_acc:
            #
            #         # found a model with better validation set accuracy
            #
            #         best_dev_acc = dev_acc
            #         snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
            #         snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc,
            #                                                                                            dev_loss.item(),
            #                                                                                            iterations)
            #
            #         # save model, delete previous 'best_snapshot' files
            #         torch.save(model, snapshot_path)
            #         for f in glob.glob(snapshot_prefix + '*'):
            #             if f != snapshot_path:
            #
            #             os.remove(f)

        # End of each epoch
        print("{}\t{}".format(total_loss, train_acc))
