import torch
import torch.utils.data


class Train:

    def __init__(self):
        pass

    def __call__(self, data_iter, model_network, loss_function, optimizer, epoch=10, mini_batch_size=1):
        """
Runs train...
        :param data_iter: An iterable record of tuples of the fomw (label, features).  The each feature must be the index of word vocab
        :param model_network: A neural network
        :param loss_function: Pytorch loss function
        :param optimizer: Optimiser
        """
        losses = []
        train_loader = torch.utils.data.DataLoader(data_iter, batch_size=mini_batch_size, shuffle=True, )
        n_correct, n_total = 0, 0
        for epoch in range(epoch):
            total_loss = 0
            for batch_y, batch_x in train_loader:
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




            # End of each epoch
            print("{}\t{}".format(total_loss, train_acc))
